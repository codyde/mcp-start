# mcp-tanstack-start

MCP (Model Context Protocol) integration for [TanStack Start](https://tanstack.com/start). Build AI-powered tools that can be called by LLMs using the standardized MCP protocol.

## Installation

```bash
npm install mcp-tanstack-start @modelcontextprotocol/sdk zod
```

or with your preferred package manager:

```bash
pnpm add mcp-tanstack-start @modelcontextprotocol/sdk zod
yarn add mcp-tanstack-start @modelcontextprotocol/sdk zod
```

## Quick Start

Get up and running with a single file. Here's a complete MCP server with tools in one API route:

```typescript
// src/routes/api/mcp.ts
import { createFileRoute } from '@tanstack/react-router'
import { createMcpServer, defineTool } from 'mcp-tanstack-start'
import { z } from 'zod'

// Define a tool
const echoTool = defineTool({
  name: 'echo',
  description: 'Echo back a message',
  parameters: z.object({
    message: z.string().describe('The message to echo back'),
  }),
  execute: async ({ message }) => {
    return `You said: ${message}`
  },
})

// Create the MCP server
const mcp = createMcpServer({
  name: 'my-tanstack-app',
  version: '1.0.0',
  instructions: `This is my TanStack Start app with MCP tools.
You can use the available tools to interact with the application.`,
  tools: [echoTool],
})

// Wire up the route handlers
export const Route = createFileRoute('/api/mcp')({
  server: {
    handlers: {
      POST: async ({ request }) => {
        return mcp.handleRequest(request)
      },
      GET: async ({ request }) => {
        return mcp.handleRequest(request)
      },
    },
  },
})
```

That's it! Your MCP server is now live at `/api/mcp`.

## Breaking It Down

### Setting Up the API Route

The API route is where your MCP server lives. It handles both POST (for JSON-RPC requests) and GET (for SSE streams):

```typescript
// src/routes/api/mcp.ts
import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/api/mcp')({
  server: {
    handlers: {
      POST: async ({ request }) => {
        return mcp.handleRequest(request)
      },
      GET: async ({ request }) => {
        return mcp.handleRequest(request)
      },
    },
  },
})
```

### Creating the MCP Server

The MCP server manages your tools and handles the protocol communication:

```typescript
const mcp = createMcpServer({
  name: 'my-tanstack-app',      // Server name
  version: '1.0.0',              // Server version
  instructions: `Optional instructions for AI assistants about how to use your tools.`,
  tools: [echoTool],             // Array of tools
})
```

### Defining Tools

Tools are the functions that LLMs can call. Each tool has a name, description, parameters (defined with Zod), and an execute function:

```typescript
import { defineTool } from 'mcp-tanstack-start'
import { z } from 'zod'

const echoTool = defineTool({
  name: 'echo',
  description: 'Echo back a message',
  parameters: z.object({
    message: z.string().describe('The message to echo back'),
  }),
  execute: async ({ message }) => {
    return `You said: ${message}`
  },
})
```

The `parameters` object uses Zod schemas for type-safe validation. The `execute` function receives the validated parameters and returns a string response.

## Authentication

Protect your MCP endpoint with authentication:

```typescript
// src/routes/api/mcp.ts
import { createFileRoute } from '@tanstack/react-router'
import { withMcpAuth } from 'mcp-tanstack-start'
import { mcp } from '../../mcp'
import { verifyJWT } from '../../lib/auth'

const authenticatedHandler = withMcpAuth(
  async (request, auth) => {
    return mcp.handleRequest(request, { auth })
  },
  async (request) => {
    const token = request.headers.get('Authorization')?.replace('Bearer ', '')
    if (!token) return null
    try {
      const claims = await verifyJWT(token)
      return { token, claims }
    } catch {
      return null
    }
  }
)

export const Route = createFileRoute('/api/mcp')({
  server: {
    handlers: {
      POST: authenticatedHandler,
      GET: authenticatedHandler,
    },
  },
})
```

Access auth in tools:

```typescript
const userDataTool = defineTool({
  name: 'get_user_data',
  description: 'Get data for the authenticated user',
  parameters: z.object({}),
  execute: async (params, context) => {
    const userId = context.auth?.claims?.sub
    if (!userId) {
      return { content: [{ type: 'text', text: 'Not authenticated' }], isError: true }
    }
    const userData = await fetchUserData(userId)
    return JSON.stringify(userData)
  },
})
```

## API Reference

### `createMcpServer(config)`

Creates an MCP server instance.

```typescript
const mcp = createMcpServer({
  name: string,           // Server name
  version: string,        // Server version
  tools?: ToolDefinition[], // Array of tools
  instructions?: string,  // Optional instructions for AI
})

// Returns
mcp.handleRequest(request: Request, options?: { auth?, signal? }): Promise<Response>
mcp.addTool(tool: ToolDefinition): void
mcp.getInfo(): { name: string; version: string }
```

### `defineTool(config)`

Defines a tool with type-safe parameters.

```typescript
defineTool({
  name: string,
  description: string,
  parameters: ZodSchema,
  execute: (params, context) => Promise<string | Content[] | ToolResult>
})
```

### `withMcpAuth(handler, verifyToken, options?)`

Wraps a handler with authentication.

```typescript
withMcpAuth(handler, verifyToken, {
  realm?: string,              // WWW-Authenticate realm
  requiredScopes?: string[],   // Required scopes
  allowUnauthenticated?: boolean,
})
```

### Content Helpers

- `text(content: string)` - Create text content
- `image(data: string, mimeType: string)` - Create image content (base64)
- `resource(uri: string, options?)` - Create embedded resource

## Protocol

Implements [MCP Streamable HTTP transport](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports) in stateless mode:

- **POST /api/mcp** - Main endpoint for MCP operations
- **GET /api/mcp** - SSE stream for server-to-client notifications
- JSON-RPC 2.0 message format

Supported methods: `initialize`, `initialized`, `tools/list`, `tools/call`, `ping`

## Examples

Check out the [example blog implementation](https://github.com/codyde/codyde-start) to see mcp-tanstack-start in action with:
- Blog post listing and retrieval
- Content search
- Server info tools

## License

MIT
