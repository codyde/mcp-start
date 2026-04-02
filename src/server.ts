import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import type { z } from "zod";
import { WebStandardTransport } from "./web-transport.js";
import type {
  McpServerConfig,
  McpServerInstance,
  McpRequestOptions,
  ToolDefinition,
  ToolExecutionContext,
} from "./types.js";

/**
 * Create an MCP server instance for use with TanStack Start.
 *
 * @param config - Server configuration including name, version, tools, and transport options
 */
export function createMcpServer(config: McpServerConfig): McpServerInstance {
  const { name, version, tools = [], instructions, transport: transportOptions } = config;

  // Store tool definitions for registration
  const toolDefinitions: ToolDefinition[] = [...tools];

  // Create the underlying MCP server
  const server = new McpServer(
    { name, version },
    {
      capabilities: {
        tools: toolDefinitions.length > 0 ? {} : undefined,
      },
      instructions,
    }
  );

  // Create our Web Standard transport with optional configuration
  const transport = new WebStandardTransport(transportOptions);

  // Register all tools with access to transport for auth context
  for (const tool of toolDefinitions) {
    registerTool(server, tool, transport);
  }

  // Connection state with proper locking to prevent race conditions
  let connectionPromise: Promise<void> | null = null;

  return {
    handleRequest: async (
      request: Request,
      options?: McpRequestOptions
    ): Promise<Response> => {
      // Ensure server is connected to transport (with race condition protection)
      if (!connectionPromise) {
        connectionPromise = server.connect(transport);
      }
      await connectionPromise;

      return transport.handleRequest(request, options);
    },

    addTool: (tool: ToolDefinition) => {
      toolDefinitions.push(tool);
      registerTool(server, tool, transport);
    },

    getInfo: () => ({ name, version }),
  };
}

function isZodSchemaLike(value: unknown): value is z.ZodType {
  return (
    typeof value === "object" &&
    value !== null &&
    ("_def" in value ||
      "_zod" in value ||
      typeof (value as { safeParse?: unknown }).safeParse === "function")
  );
}

/**
 * Plain object whose values are Zod field schemas — the MCP SDK accepts this as
 * a tool input shape (same as `z.object({ ... }).shape`).
 */
function isZodRawShape(value: unknown): value is Record<string, z.ZodType> {
  if (!value || typeof value !== "object" || isZodSchemaLike(value)) {
    return false;
  }
  const vals = Object.values(value);
  if (vals.length === 0) return false;
  return vals.every(isZodSchemaLike);
}

/**
 * Extract raw shape for MCP tool registration from:
 * - A Zod object schema (`z.object({ ... })`)
 * - A raw shape object (`{ foo: z.string() }`)
 *
 * Covers Zod 3 (`_def.typeName === "ZodObject"`), Zod 4 classic-style objects,
 * and bundlers where only the public `.shape` getter is reliable.
 */
function extractZodShape(schema: unknown): Record<string, z.ZodType> | undefined {
  if (isZodRawShape(schema)) {
    return schema;
  }

  if (!isZodSchemaLike(schema)) {
    return undefined;
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const def = (schema as z.ZodType & { shape?: Record<string, z.ZodType> })._def as any;

  if (def?.typeName === "ZodObject" && typeof def.shape === "function") {
    return def.shape() as Record<string, z.ZodType>;
  }

  if (def?.type === "object") {
    const sh = def.shape;
    if (typeof sh === "function") return sh() as Record<string, z.ZodType>;
    if (sh && typeof sh === "object") return sh as Record<string, z.ZodType>;
  }

  const { shape } = schema as z.ZodType & { shape?: unknown };
  if (shape && typeof shape === "object") {
    const vals = Object.values(shape as Record<string, unknown>);
    if (
      vals.some(
        (v) =>
          v &&
          typeof v === "object" &&
          isZodSchemaLike(v)
      )
    ) {
      return shape as Record<string, z.ZodType>;
    }
  }

  return undefined;
}

/**
 * Register a tool with the MCP server
 */
function registerTool(
  server: McpServer,
  tool: ToolDefinition,
  transport: WebStandardTransport
): void {
  // Extract the raw shape from the Zod object schema
  // The SDK expects a raw shape like { message: z.string() }, not z.object({...})
  const shape = extractZodShape(tool.parameters);

  const executeWithContext = async (
    params: Record<string, unknown>
  ): Promise<CallToolResult> => {
    // Get auth and signal from the transport's current request options
    const options = transport.getCurrentOptions();

    const context: ToolExecutionContext = {
      auth: options?.auth,
      signal: options?.signal,
    };

    try {
      const result = await tool.execute(params, context);
      return normalizeToCallToolResult(result);
    } catch (error) {
      // Sanitize error message to avoid leaking sensitive information
      const message =
        error instanceof Error
          ? error.message
          : "An error occurred during tool execution";
      return {
        content: [{ type: "text", text: `Error: ${message}` }],
        isError: true,
      };
    }
  };

  if (shape) {
    // Use the deprecated .tool() method with the raw shape
    // Note: The SDK's current API requires this approach for custom parameter shapes.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (server as any).tool(
      tool.name,
      tool.description,
      shape,
      async (params: Record<string, unknown>): Promise<CallToolResult> => {
        return executeWithContext(params);
      }
    );
  } else {
    // Fallback for non-object schemas - use empty shape
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (server as any).tool(
      tool.name,
      tool.description,
      {},
      async (): Promise<CallToolResult> => {
        return executeWithContext({});
      }
    );
  }
}

/**
 * Normalize tool execution result to SDK's CallToolResult format
 */
function normalizeToCallToolResult(result: unknown): CallToolResult {
  if (typeof result === "string") {
    return { content: [{ type: "text", text: result }] };
  }

  if (Array.isArray(result)) {
    return { content: result.map(normalizeContentItem) };
  }

  if (
    typeof result === "object" &&
    result !== null &&
    "content" in result &&
    Array.isArray((result as { content: unknown[] }).content)
  ) {
    const typedResult = result as { content: unknown[]; isError?: boolean };
    return {
      content: typedResult.content.map(normalizeContentItem),
      isError: typedResult.isError,
    };
  }

  return { content: [{ type: "text", text: JSON.stringify(result) }] };
}

function normalizeContentItem(
  item: unknown
): CallToolResult["content"][number] {
  if (typeof item === "string") {
    return { type: "text", text: item };
  }

  if (typeof item === "object" && item !== null && "type" in item) {
    const typed = item as { type: string; [key: string]: unknown };

    switch (typed.type) {
      case "text":
        return { type: "text", text: String(typed.text || "") };
      case "image":
        return {
          type: "image",
          data: String(typed.data || ""),
          mimeType: String(typed.mimeType || "image/png"),
        };
      case "resource": {
        const resource = typed.resource as {
          uri: string;
          text?: string;
          blob?: string;
          mimeType?: string;
        };
        if (resource.text) {
          return {
            type: "resource",
            resource: { uri: resource.uri, text: resource.text, mimeType: resource.mimeType },
          };
        }
        if (resource.blob) {
          return {
            type: "resource",
            resource: { uri: resource.uri, blob: resource.blob, mimeType: resource.mimeType },
          };
        }
        return {
          type: "resource",
          resource: { uri: resource.uri, text: "" },
        };
      }
      default:
        return { type: "text", text: JSON.stringify(item) };
    }
  }

  return { type: "text", text: JSON.stringify(item) };
}
