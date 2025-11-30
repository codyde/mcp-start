import type { Transport } from "@modelcontextprotocol/sdk/shared/transport.js";
import type { JSONRPCMessage } from "@modelcontextprotocol/sdk/types.js";
import {
  JSONRPCMessageSchema,
  isJSONRPCRequest,
  isJSONRPCResponse,
  isJSONRPCError,
  isInitializeRequest,
  SUPPORTED_PROTOCOL_VERSIONS,
} from "@modelcontextprotocol/sdk/types.js";
import type { McpRequestOptions } from "./types.js";

/**
 * Configuration options for WebStandardTransport
 */
export interface WebStandardTransportOptions {
  /**
   * Whether to enable JSON responses instead of SSE for POST requests.
   * Default: false (uses SSE)
   */
  enableJsonResponse?: boolean;

  /**
   * Maximum request body size in bytes.
   * Default: 1MB (1048576 bytes)
   */
  maxBodySize?: number;

  /**
   * Maximum number of messages allowed in a batch request.
   * Default: 100
   */
  maxBatchSize?: number;

  /**
   * Request timeout in milliseconds.
   * Default: 30000 (30 seconds)
   */
  requestTimeout?: number;

  /**
   * Session timeout in milliseconds. Sessions inactive for this duration will be cleaned up.
   * Default: 300000 (5 minutes)
   */
  sessionTimeout?: number;
}

/**
 * Tracks a batch of requests waiting for responses
 */
interface PendingBatch {
  requestIds: Set<string | number>;
  resolve: (response: Response) => void;
  responses: Map<string | number, JSONRPCMessage>;
  expectedCount: number;
  timeoutId: ReturnType<typeof setTimeout>;
  resolved: boolean;
  sessionId: string;
}

/**
 * Session state for a single MCP client session
 */
interface Session {
  id: string;
  initialized: boolean;
  initializing: boolean;
  sseController: ReadableStreamDefaultController<Uint8Array> | null;
  sseStreamActive: boolean;
  pendingBatches: Map<string, PendingBatch>;
  requestToBatch: Map<string | number, string>;
  lastActivity: number;
  timeoutId?: ReturnType<typeof setTimeout>;
}

/**
 * Generate a cryptographically secure session ID
 */
function generateSessionId(): string {
  // Use crypto.randomUUID if available (modern runtimes)
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  // Fallback for older environments
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}-${Math.random().toString(36).slice(2)}`;
}

/**
 * Web Standard Transport for MCP
 *
 * This transport implements the MCP Streamable HTTP specification using
 * Web Standard APIs (Request/Response) instead of Node.js http module.
 *
 * Supports proper session management via Mcp-Session-Id header as per the spec:
 * https://modelcontextprotocol.io/specification/2025-06-18/basic/transports
 *
 * Designed for modern JavaScript runtimes and frameworks like:
 * - TanStack Start
 * - Remix
 * - Next.js (App Router)
 * - Cloudflare Workers
 * - Deno
 * - Bun
 */
export class WebStandardTransport implements Transport {
  private _started = false;
  private _enableJsonResponse: boolean;
  private _maxBodySize: number;
  private _maxBatchSize: number;
  private _requestTimeout: number;
  private _sessionTimeout: number;

  // Current request options (auth, signal, etc.)
  private _currentOptions?: McpRequestOptions;

  // Session management - keyed by session ID
  private _sessions = new Map<string, Session>();

  // Current session ID being processed (set during request handling)
  private _currentSessionId?: string;

  // Transport callbacks (set by the SDK when connecting)
  onclose?: () => void;
  onerror?: (error: Error) => void;
  onmessage?: (message: JSONRPCMessage) => void;

  constructor(options: WebStandardTransportOptions = {}) {
    this._enableJsonResponse = options.enableJsonResponse ?? false;
    this._maxBodySize = options.maxBodySize ?? 1048576; // 1MB default
    this._maxBatchSize = options.maxBatchSize ?? 100;
    this._requestTimeout = options.requestTimeout ?? 30000; // 30 seconds default
    this._sessionTimeout = options.sessionTimeout ?? 300000; // 5 minutes default
  }

  /**
   * Get the current request options (auth info, signal, etc.)
   */
  getCurrentOptions(): McpRequestOptions | undefined {
    return this._currentOptions;
  }

  /**
   * Start the transport. Required by Transport interface.
   */
  async start(): Promise<void> {
    if (this._started) {
      throw new Error("Transport already started");
    }
    this._started = true;
  }

  /**
   * Close the transport and clean up all sessions.
   */
  async close(): Promise<void> {
    // Clean up all sessions
    for (const session of this._sessions.values()) {
      this.cleanupSession(session);
    }
    this._sessions.clear();

    this.onclose?.();
  }

  /**
   * Create a new session
   */
  private createSession(): Session {
    const id = generateSessionId();
    const session: Session = {
      id,
      initialized: false,
      initializing: false,
      sseController: null,
      sseStreamActive: false,
      pendingBatches: new Map(),
      requestToBatch: new Map(),
      lastActivity: Date.now(),
    };

    // Set up session timeout
    session.timeoutId = setTimeout(() => {
      this.terminateSession(id);
    }, this._sessionTimeout);

    this._sessions.set(id, session);
    return session;
  }

  /**
   * Get a session by ID, or return null if not found
   */
  private getSession(sessionId: string): Session | null {
    const session = this._sessions.get(sessionId);
    if (session) {
      // Reset the session timeout on activity
      this.refreshSessionTimeout(session);
    }
    return session ?? null;
  }

  /**
   * Refresh the session timeout
   */
  private refreshSessionTimeout(session: Session): void {
    session.lastActivity = Date.now();
    if (session.timeoutId) {
      clearTimeout(session.timeoutId);
    }
    session.timeoutId = setTimeout(() => {
      this.terminateSession(session.id);
    }, this._sessionTimeout);
  }

  /**
   * Terminate a session and clean up resources
   */
  terminateSession(sessionId: string): boolean {
    const session = this._sessions.get(sessionId);
    if (!session) {
      return false;
    }

    this.cleanupSession(session);
    this._sessions.delete(sessionId);
    return true;
  }

  /**
   * Clean up session resources
   */
  private cleanupSession(session: Session): void {
    // Clear timeout
    if (session.timeoutId) {
      clearTimeout(session.timeoutId);
    }

    // Close SSE stream
    if (session.sseController) {
      try {
        session.sseController.close();
      } catch {
        // May already be closed
      }
      session.sseController = null;
    }
    session.sseStreamActive = false;

    // Reject any pending batches
    for (const batch of session.pendingBatches.values()) {
      if (!batch.resolved) {
        clearTimeout(batch.timeoutId);
        batch.resolved = true;
        batch.resolve(
          new Response(
            JSON.stringify({
              jsonrpc: "2.0",
              error: { code: -32000, message: "Session terminated" },
              id: null,
            }),
            { status: 500, headers: { "Content-Type": "application/json" } }
          )
        );
      }
    }
    session.pendingBatches.clear();
    session.requestToBatch.clear();
  }

  /**
   * Clean up SSE stream for a session
   */
  private cleanupSseStream(session: Session): void {
    if (session.sseController) {
      try {
        session.sseController.close();
      } catch {
        // May already be closed
      }
      session.sseController = null;
    }
    session.sseStreamActive = false;
  }

  /**
   * Send a message (response or notification) back to the client.
   * Called by the MCP server when it has a response ready.
   */
  async send(message: JSONRPCMessage): Promise<void> {
    // If it's a response/error, find the pending batch and add the response
    if (isJSONRPCResponse(message) || isJSONRPCError(message)) {
      const requestId = message.id;

      // Search all sessions for this request
      for (const session of this._sessions.values()) {
        const batchId = session.requestToBatch.get(requestId);
        if (batchId) {
          const batch = session.pendingBatches.get(batchId);
          if (batch && !batch.resolved) {
            batch.responses.set(requestId, message);

            // Check if we have all expected responses
            if (batch.responses.size >= batch.expectedCount) {
              this.resolveBatch(session, batchId, batch);
            }
          }
          return;
        }
      }
      return;
    }

    // For notifications/requests from server, send on SSE stream if available
    // Use current session if set, otherwise broadcast to all active streams
    if (this._currentSessionId) {
      const session = this._sessions.get(this._currentSessionId);
      if (session?.sseController && session.sseStreamActive) {
        try {
          const sseEvent = `event: message\ndata: ${JSON.stringify(message)}\n\n`;
          session.sseController.enqueue(new TextEncoder().encode(sseEvent));
        } catch {
          // Stream may have been closed
        }
      }
    }
  }

  /**
   * Resolve a pending batch with all its responses
   */
  private resolveBatch(session: Session, batchId: string, batch: PendingBatch): void {
    if (batch.resolved) return;

    batch.resolved = true;
    clearTimeout(batch.timeoutId);

    // Clean up mappings
    for (const reqId of batch.requestIds) {
      session.requestToBatch.delete(reqId);
    }
    session.pendingBatches.delete(batchId);

    // Build response array in original request order
    const responses: JSONRPCMessage[] = [];
    for (const reqId of batch.requestIds) {
      const response = batch.responses.get(reqId);
      if (response) {
        responses.push(response);
      }
    }

    // Add session ID header to response
    const headers: Record<string, string> = {};
    if (batch.sessionId) {
      headers["Mcp-Session-Id"] = batch.sessionId;
    }

    if (this._enableJsonResponse) {
      // Return as JSON
      const body =
        responses.length === 1
          ? JSON.stringify(responses[0])
          : JSON.stringify(responses);

      batch.resolve(
        new Response(body, {
          status: 200,
          headers: { ...headers, "Content-Type": "application/json" },
        })
      );
    } else {
      // Return as SSE
      const sseData = responses
        .map((r) => `event: message\ndata: ${JSON.stringify(r)}\n\n`)
        .join("");

      batch.resolve(
        new Response(sseData, {
          status: 200,
          headers: {
            ...headers,
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            Connection: "keep-alive",
          },
        })
      );
    }
  }

  /**
   * Handle an incoming HTTP request.
   * This is the main entry point for the transport.
   */
  async handleRequest(request: Request, options?: McpRequestOptions): Promise<Response> {
    // Store options for access during tool execution
    this._currentOptions = options;

    try {
      if (request.method === "GET") {
        return await this.handleGetRequest(request);
      }

      if (request.method === "POST") {
        return await this.handlePostRequest(request);
      }

      if (request.method === "DELETE") {
        return await this.handleDeleteRequest(request);
      }

      return new Response(
        JSON.stringify({
          jsonrpc: "2.0",
          error: { code: -32000, message: "Method not allowed. Use GET, POST, or DELETE." },
          id: null,
        }),
        {
          status: 405,
          headers: { "Content-Type": "application/json", Allow: "GET, POST, DELETE" },
        }
      );
    } finally {
      // Clear options after request is handled
      this._currentOptions = undefined;
      this._currentSessionId = undefined;
    }
  }

  /**
   * Handle DELETE requests to terminate a session
   */
  private async handleDeleteRequest(request: Request): Promise<Response> {
    const sessionId = request.headers.get("mcp-session-id");

    if (!sessionId) {
      return new Response(
        JSON.stringify({
          jsonrpc: "2.0",
          error: { code: -32000, message: "Bad Request: Missing Mcp-Session-Id header" },
          id: null,
        }),
        { status: 400, headers: { "Content-Type": "application/json" } }
      );
    }

    const terminated = this.terminateSession(sessionId);

    if (!terminated) {
      return new Response(
        JSON.stringify({
          jsonrpc: "2.0",
          error: { code: -32000, message: "Not Found: Session does not exist" },
          id: null,
        }),
        { status: 404, headers: { "Content-Type": "application/json" } }
      );
    }

    return new Response(null, { status: 204 });
  }

  /**
   * Handle GET requests for SSE stream (server-to-client notifications)
   */
  private async handleGetRequest(request: Request): Promise<Response> {
    const acceptHeader = request.headers.get("accept") || "";
    const sessionId = request.headers.get("mcp-session-id");

    // Must accept text/event-stream
    if (!this.acceptsMediaType(acceptHeader, "text/event-stream")) {
      return new Response(
        JSON.stringify({
          jsonrpc: "2.0",
          error: {
            code: -32000,
            message: "Not Acceptable: Client must accept text/event-stream",
          },
          id: null,
        }),
        { status: 406, headers: { "Content-Type": "application/json" } }
      );
    }

    // Session ID is required for GET requests (after initialization)
    if (!sessionId) {
      return new Response(
        JSON.stringify({
          jsonrpc: "2.0",
          error: {
            code: -32000,
            message: "Bad Request: Missing Mcp-Session-Id header",
          },
          id: null,
        }),
        { status: 400, headers: { "Content-Type": "application/json" } }
      );
    }

    const session = this.getSession(sessionId);
    if (!session) {
      return new Response(
        JSON.stringify({
          jsonrpc: "2.0",
          error: {
            code: -32000,
            message: "Not Found: Session does not exist or has expired",
          },
          id: null,
        }),
        { status: 404, headers: { "Content-Type": "application/json" } }
      );
    }

    // Set current session for message routing
    this._currentSessionId = sessionId;

    // If there's already an active SSE stream for this session, close it and start fresh
    // This handles reconnection scenarios
    if (session.sseStreamActive) {
      this.cleanupSseStream(session);
    }

    // Create SSE stream
    const stream = new ReadableStream<Uint8Array>({
      start: (controller) => {
        session.sseController = controller;
        session.sseStreamActive = true;
      },
      cancel: () => {
        this.cleanupSseStream(session);
      },
    });

    // Handle client disconnect via abort signal
    request.signal.addEventListener(
      "abort",
      () => {
        this.cleanupSseStream(session);
      },
      { once: true }
    );

    return new Response(stream, {
      status: 200,
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-transform",
        Connection: "keep-alive",
        "Mcp-Session-Id": sessionId,
      },
    });
  }

  /**
   * Check if an Accept header includes a specific media type
   */
  private acceptsMediaType(acceptHeader: string, mediaType: string): boolean {
    // Parse Accept header properly
    const parts = acceptHeader.split(",").map((p) => p.trim().split(";")[0].trim());
    return parts.some(
      (p) => p === mediaType || p === "*/*" || p === mediaType.split("/")[0] + "/*"
    );
  }

  /**
   * Handle POST requests containing JSON-RPC messages
   */
  private async handlePostRequest(request: Request): Promise<Response> {
    // Validate Accept header - must accept at least one of JSON or SSE
    const acceptHeader = request.headers.get("accept") || "";
    const acceptsJson = this.acceptsMediaType(acceptHeader, "application/json");
    const acceptsSse = this.acceptsMediaType(acceptHeader, "text/event-stream");

    if (!acceptsJson && !acceptsSse) {
      return new Response(
        JSON.stringify({
          jsonrpc: "2.0",
          error: {
            code: -32000,
            message:
              "Not Acceptable: Client must accept application/json or text/event-stream",
          },
          id: null,
        }),
        { status: 406, headers: { "Content-Type": "application/json" } }
      );
    }

    // Validate Content-Type
    const contentType = request.headers.get("content-type") || "";
    if (!contentType.toLowerCase().includes("application/json")) {
      return new Response(
        JSON.stringify({
          jsonrpc: "2.0",
          error: {
            code: -32000,
            message: "Unsupported Media Type: Content-Type must be application/json",
          },
          id: null,
        }),
        { status: 415, headers: { "Content-Type": "application/json" } }
      );
    }

    // Check Content-Length if available
    const contentLength = request.headers.get("content-length");
    if (contentLength && parseInt(contentLength, 10) > this._maxBodySize) {
      return new Response(
        JSON.stringify({
          jsonrpc: "2.0",
          error: {
            code: -32000,
            message: `Payload Too Large: Maximum body size is ${this._maxBodySize} bytes`,
          },
          id: null,
        }),
        { status: 413, headers: { "Content-Type": "application/json" } }
      );
    }

    // Parse the request body
    let rawMessage: unknown;
    try {
      const text = await request.text();

      // Check actual body size
      if (text.length > this._maxBodySize) {
        return new Response(
          JSON.stringify({
            jsonrpc: "2.0",
            error: {
              code: -32000,
              message: `Payload Too Large: Maximum body size is ${this._maxBodySize} bytes`,
            },
            id: null,
          }),
          { status: 413, headers: { "Content-Type": "application/json" } }
        );
      }

      rawMessage = JSON.parse(text);
    } catch {
      return new Response(
        JSON.stringify({
          jsonrpc: "2.0",
          error: { code: -32700, message: "Parse error: Invalid JSON" },
          id: null,
        }),
        { status: 400, headers: { "Content-Type": "application/json" } }
      );
    }

    // Check batch size limit
    if (Array.isArray(rawMessage) && rawMessage.length > this._maxBatchSize) {
      return new Response(
        JSON.stringify({
          jsonrpc: "2.0",
          error: {
            code: -32000,
            message: `Batch Too Large: Maximum ${this._maxBatchSize} messages per batch`,
          },
          id: null,
        }),
        { status: 400, headers: { "Content-Type": "application/json" } }
      );
    }

    // Parse and validate JSON-RPC messages
    let messages: JSONRPCMessage[];
    try {
      if (Array.isArray(rawMessage)) {
        messages = rawMessage.map((msg) => JSONRPCMessageSchema.parse(msg));
      } else {
        messages = [JSONRPCMessageSchema.parse(rawMessage)];
      }
    } catch {
      return new Response(
        JSON.stringify({
          jsonrpc: "2.0",
          error: {
            code: -32600,
            message: "Invalid Request: Not a valid JSON-RPC message",
          },
          id: null,
        }),
        { status: 400, headers: { "Content-Type": "application/json" } }
      );
    }

    // Check for initialization request
    const isInitializationRequest = messages.some(isInitializeRequest);
    const requestSessionId = request.headers.get("mcp-session-id");

    let session: Session;

    if (isInitializationRequest) {
      // Only allow single initialization request
      if (messages.length > 1) {
        return new Response(
          JSON.stringify({
            jsonrpc: "2.0",
            error: {
              code: -32600,
              message: "Invalid Request: Only one initialization request is allowed",
            },
            id: null,
          }),
          { status: 400, headers: { "Content-Type": "application/json" } }
        );
      }

      // Create a new session for initialization
      // If there was an old session ID provided, terminate it first
      if (requestSessionId) {
        this.terminateSession(requestSessionId);
      }

      session = this.createSession();
      session.initializing = true;
    } else {
      // Non-initialization requests require a session ID
      if (!requestSessionId) {
        return new Response(
          JSON.stringify({
            jsonrpc: "2.0",
            error: {
              code: -32000,
              message: "Bad Request: Missing Mcp-Session-Id header",
            },
            id: null,
          }),
          { status: 400, headers: { "Content-Type": "application/json" } }
        );
      }

      const existingSession = this.getSession(requestSessionId);
      if (!existingSession) {
        return new Response(
          JSON.stringify({
            jsonrpc: "2.0",
            error: {
              code: -32000,
              message: "Not Found: Session does not exist or has expired",
            },
            id: null,
          }),
          { status: 404, headers: { "Content-Type": "application/json" } }
        );
      }

      session = existingSession;

      // Validate protocol version for non-initialization requests
      const protocolVersion = request.headers.get("mcp-protocol-version");
      if (protocolVersion && !SUPPORTED_PROTOCOL_VERSIONS.includes(protocolVersion)) {
        return new Response(
          JSON.stringify({
            jsonrpc: "2.0",
            error: {
              code: -32000,
              message: `Bad Request: Unsupported protocol version (supported: ${SUPPORTED_PROTOCOL_VERSIONS.join(", ")})`,
            },
            id: null,
          }),
          { status: 400, headers: { "Content-Type": "application/json" } }
        );
      }
    }

    // Set current session for message routing
    this._currentSessionId = session.id;

    // Check if the batch contains requests (not just notifications)
    const requests = messages.filter(isJSONRPCRequest);
    const hasRequests = requests.length > 0;

    if (!hasRequests) {
      // Only notifications - process and return 202 Accepted
      for (const message of messages) {
        this.onmessage?.(message);
      }

      // Mark as initialized if this was an initialization
      if (isInitializationRequest) {
        session.initialized = true;
        session.initializing = false;
      }

      return new Response(null, {
        status: 202,
        headers: { "Mcp-Session-Id": session.id },
      });
    }

    // Create a promise that will resolve when we have all responses
    return new Promise<Response>((resolve) => {
      // Generate unique batch ID
      const batchId = `batch_${Date.now()}_${Math.random().toString(36).slice(2)}`;

      // Set up timeout
      const timeoutId = setTimeout(() => {
        const batch = session.pendingBatches.get(batchId);
        if (batch && !batch.resolved) {
          batch.resolved = true;

          // Clean up mappings
          for (const reqId of batch.requestIds) {
            session.requestToBatch.delete(reqId);
          }
          session.pendingBatches.delete(batchId);

          // Build timeout response including any responses we did receive
          const responses: JSONRPCMessage[] = [];
          for (const reqId of batch.requestIds) {
            const existingResponse = batch.responses.get(reqId);
            if (existingResponse) {
              responses.push(existingResponse);
            } else {
              // Create timeout error for missing responses
              responses.push({
                jsonrpc: "2.0",
                error: { code: -32001, message: "Request timed out" },
                id: reqId,
              } as JSONRPCMessage);
            }
          }

          const body =
            responses.length === 1
              ? JSON.stringify(responses[0])
              : JSON.stringify(responses);

          resolve(
            new Response(body, {
              status: 408,
              headers: {
                "Content-Type": "application/json",
                "Mcp-Session-Id": session.id,
              },
            })
          );
        }
      }, this._requestTimeout);

      // Track the batch
      const requestIds = new Set(requests.map((r) => r.id));
      const batch: PendingBatch = {
        requestIds,
        resolve,
        responses: new Map(),
        expectedCount: requests.length,
        timeoutId,
        resolved: false,
        sessionId: session.id,
      };

      session.pendingBatches.set(batchId, batch);

      // Map each request ID to this batch
      for (const reqId of requestIds) {
        session.requestToBatch.set(reqId, batchId);
      }

      // Dispatch messages to the MCP server
      for (const message of messages) {
        this.onmessage?.(message);
      }

      // Mark as initialized after dispatching if this was initialization
      if (isInitializationRequest) {
        session.initialized = true;
        session.initializing = false;
      }
    });
  }

  /**
   * Reset session state for a new connection (deprecated - use session management instead).
   * Kept for backwards compatibility but now just cleans up all sessions.
   */
  resetSession(): void {
    for (const session of this._sessions.values()) {
      this.cleanupSession(session);
    }
    this._sessions.clear();
  }
}
