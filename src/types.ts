import type { z } from "zod";

/**
 * Content types that can be returned from tool execution
 */
export interface TextContent {
  type: "text";
  text: string;
}

export interface ImageContent {
  type: "image";
  data: string;
  mimeType: string;
}

export interface EmbeddedResource {
  type: "resource";
  resource: {
    uri: string;
    mimeType?: string;
    text?: string;
    blob?: string;
  };
}

export type ToolContent = TextContent | ImageContent | EmbeddedResource;

/**
 * Result returned from tool execution
 */
export interface ToolResult {
  content: ToolContent[];
  isError?: boolean;
}

/**
 * Tool definition configuration
 */
export interface ToolDefinition<TParams extends z.ZodType = z.ZodType> {
  /** Unique name for the tool */
  name: string;
  /** Human-readable description of what the tool does */
  description: string;
  /**
   * Tool parameters: typically `z.object({ ... })`, or a raw shape object
   * `{ key: z.string(), ... }` (same as `.shape` on a Zod object).
   */
  parameters: TParams;
  /** Function that executes the tool */
  execute: (
    params: z.infer<TParams>,
    context: ToolExecutionContext
  ) => Promise<string | ToolContent[] | ToolResult>;
}

/**
 * Context passed to tool execution
 */
export interface ToolExecutionContext {
  /** Authentication info if auth middleware is used */
  auth?: AuthInfo;
  /** Abort signal for cancellation */
  signal?: AbortSignal;
}

/**
 * Authentication information
 */
export interface AuthInfo {
  /** Token used for authentication */
  token: string;
  /** User-defined claims/data from token verification */
  claims?: Record<string, unknown>;
  /** Scopes granted to this token */
  scopes?: string[];
}

/**
 * Base tool definition type for arrays (non-generic for variance compatibility)
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type AnyToolDefinition = ToolDefinition<any>;

/**
 * Minimal session data that can be persisted
 */
export interface SessionData {
  /** Unique session identifier */
  id: string;
  /** Whether the session has completed initialization */
  initialized: boolean;
  /** Negotiated protocol version */
  protocolVersion: string;
  /** Timestamp of last activity (for TTL tracking) */
  lastActivity: number;
}

/**
 * Session store interface for pluggable session persistence.
 * Implement this interface to store sessions in Redis, DynamoDB, etc.
 */
export interface SessionStore {
  /**
   * Get a session by ID
   * @returns Session data or null if not found/expired
   */
  get(id: string): Promise<SessionData | null> | SessionData | null;

  /**
   * Store a session with a TTL
   * @param id - Session ID
   * @param session - Session data to store
   * @param ttlMs - Time-to-live in milliseconds
   */
  set(id: string, session: SessionData, ttlMs: number): Promise<void> | void;

  /**
   * Delete a session
   * @param id - Session ID to delete
   */
  delete(id: string): Promise<void> | void;
}

/**
 * Transport configuration options
 */
export interface TransportOptions {
  /**
   * Enable stateful session mode.
   * 
   * When false (default): Stateless mode - works on serverless, edge, and distributed environments.
   * Sessions are not persisted; if a session is not found, requests are processed gracefully.
   * Server-to-client push notifications are not available in this mode.
   * 
   * When true: Stateful mode - enables persistent sessions for SSE push notifications.
   * Requires either in-memory storage (single instance only) or a custom sessionStore.
   * 
   * Default: false (stateless)
   */
  stateful?: boolean;

  /**
   * Custom session store for persistent sessions.
   * Only used when stateful: true.
   * If not provided, uses in-memory storage (not suitable for distributed deployments).
   */
  sessionStore?: SessionStore;

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
   * Request timeout in milliseconds.
   * Default: 30000 (30 seconds)
   */
  requestTimeout?: number;

  /**
   * Session timeout in milliseconds. Sessions inactive for this duration will be cleaned up.
   * Only applies in stateful mode.
   * Default: 3600000 (1 hour)
   */
  sessionTimeout?: number;

  /**
   * Allowed origins for Origin header validation (DNS rebinding protection).
   * If not provided, defaults to allowing localhost origins only.
   * Set to ["*"] to allow all origins (NOT recommended for production).
   */
  allowedOrigins?: string[];

  /**
   * Whether to enable SSE event IDs for resumability.
   * Only applies in stateful mode.
   * Default: true
   */
  enableResumability?: boolean;
}

/**
 * MCP Server configuration
 */
export interface McpServerConfig {
  /** Name of the MCP server */
  name: string;
  /** Version of the MCP server */
  version: string;
  /** Tools to register with the server */
  tools?: AnyToolDefinition[];
  /** Optional instructions for the AI on how to use this server */
  instructions?: string;
  /** Transport configuration options */
  transport?: TransportOptions;
}

/**
 * Options for handling MCP requests
 */
export interface McpRequestOptions {
  /** Authentication info to pass to tool execution */
  auth?: AuthInfo;
  /** Abort signal for cancellation */
  signal?: AbortSignal;
}

/**
 * MCP Server instance returned by createMcpServer
 */
export interface McpServerInstance {
  /** Handle an incoming MCP request */
  handleRequest: (
    request: Request,
    options?: McpRequestOptions
  ) => Promise<Response>;
  /** Add a tool to the server */
  addTool: (tool: AnyToolDefinition) => void;
  /** Get server info */
  getInfo: () => { name: string; version: string };
}

/**
 * Token verification function for auth middleware
 */
export type TokenVerifier = (
  request: Request
) => Promise<AuthInfo | null> | AuthInfo | null;

/**
 * Authenticated request handler
 */
export type AuthenticatedHandler = (
  request: Request,
  auth: AuthInfo
) => Promise<Response>;

/**
 * JSON-RPC types for MCP protocol
 */
export interface JsonRpcRequest {
  jsonrpc: "2.0";
  id?: string | number;
  method: string;
  params?: Record<string, unknown>;
}

export interface JsonRpcResponse {
  jsonrpc: "2.0";
  id?: string | number;
  result?: unknown;
  error?: JsonRpcError;
}

export interface JsonRpcError {
  code: number;
  message: string;
  data?: unknown;
}

/**
 * MCP-specific method types
 */
export type McpMethod =
  | "initialize"
  | "initialized"
  | "tools/list"
  | "tools/call"
  | "resources/list"
  | "resources/read"
  | "prompts/list"
  | "prompts/get"
  | "ping";
