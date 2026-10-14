export interface ChatNode {
  id: string;
  parentId: string | null;
  childrenIds: string[];
  role: "user" | "assistant" | "system";
  content: string;
  createdAt: number;
}

export interface ChatSession {
  id: string;
  title: string;
  personaId: string;
  reasoningEffort?: string;
  rootNodeId: string | null;
  activeNodeId: string | null;
  nodes: Record<string, ChatNode>;
  createdAt: number;
  updatedAt: number;
}

export interface Persona {
  id: string;
  name: string;
  tagline: string;
  systemPrompt: string;
  isCustom?: boolean;
}

export const DEFAULT_PERSONAS: Persona[] = [
  {
    id: "default",
    name: "Default Assistant",
    tagline: "Basic AI assistant",
    systemPrompt: "You are a helpful, harmless, and honest AI assistant.",
  },
];

const STORAGE_KEY_SESSIONS = "llama_chat_sessions_v1";
const STORAGE_KEY_ACTIVE = "llama_active_session_id_v1";
const STORAGE_KEY_CUSTOM_PERSONAS = "llama_custom_personas_v1";

// Persona Management
export function getPersonas(): Persona[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY_CUSTOM_PERSONAS);
    if (!raw) return DEFAULT_PERSONAS;
    const custom: Persona[] = JSON.parse(raw);
    return [...DEFAULT_PERSONAS, ...custom];
  } catch {
    return DEFAULT_PERSONAS;
  }
}

export function saveCustomPersona(persona: Persona): void {
  try {
    const raw = localStorage.getItem(STORAGE_KEY_CUSTOM_PERSONAS);
    let custom: Persona[] = raw ? JSON.parse(raw) : [];
    custom = custom.filter((p) => p.id !== persona.id);
    custom.push({ ...persona, isCustom: true });
    localStorage.setItem(STORAGE_KEY_CUSTOM_PERSONAS, JSON.stringify(custom));
  } catch (err) {
    console.error("Failed to save custom persona:", err);
  }
}

export function deleteCustomPersona(id: string): void {
  try {
    const raw = localStorage.getItem(STORAGE_KEY_CUSTOM_PERSONAS);
    if (!raw) return;
    let custom: Persona[] = JSON.parse(raw);
    custom = custom.filter((p) => p.id !== id);
    localStorage.setItem(STORAGE_KEY_CUSTOM_PERSONAS, JSON.stringify(custom));
  } catch (err) {
    console.error("Failed to delete custom persona:", err);
  }
}

export function getPersonaById(id: string): Persona {
  const personas = getPersonas();
  return personas.find((p) => p.id === id) || DEFAULT_PERSONAS[0]!;
}

// Session Management
export function loadAllSessions(): ChatSession[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY_SESSIONS);
    if (!raw) return [];
    return JSON.parse(raw);
  } catch {
    return [];
  }
}

export function saveSession(session: ChatSession): void {
  try {
    const sessions = loadAllSessions().filter((s) => s.id !== session.id);
    sessions.unshift({ ...session, updatedAt: Date.now() });
    localStorage.setItem(STORAGE_KEY_SESSIONS, JSON.stringify(sessions));
  } catch (err) {
    console.error("Failed to save session:", err);
  }
}

export function deleteSession(id: string): void {
  try {
    const sessions = loadAllSessions().filter((s) => s.id !== id);
    localStorage.setItem(STORAGE_KEY_SESSIONS, JSON.stringify(sessions));
    if (getActiveSessionId() === id) {
      if (sessions.length > 0 && sessions[0]) {
        setActiveSessionId(sessions[0].id);
      } else {
        localStorage.removeItem(STORAGE_KEY_ACTIVE);
      }
    }
  } catch (err) {
    console.error("Failed to delete session:", err);
  }
}

export function getActiveSessionId(): string | null {
  return localStorage.getItem(STORAGE_KEY_ACTIVE);
}

export function setActiveSessionId(id: string): void {
  localStorage.setItem(STORAGE_KEY_ACTIVE, id);
}

export function createNewSession(personaId = "default", reasoningEffort = "default"): ChatSession {
  const id = "sess_" + Date.now() + "_" + Math.random().toString(36).slice(2, 7);
  const session: ChatSession = {
    id,
    title: "New Conversation",
    personaId,
    reasoningEffort,
    rootNodeId: null,
    activeNodeId: null,
    nodes: {},
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };
  saveSession(session);
  setActiveSessionId(id);
  return session;
}

// Tree Node Operations
function generateNodeId(): string {
  return "node_" + Date.now() + "_" + Math.random().toString(36).slice(2, 7);
}

export function addMessageToSession(
  session: ChatSession,
  role: "user" | "assistant" | "system",
  content: string
): ChatNode {
  const id = generateNodeId();
  const parentId = session.activeNodeId;

  const node: ChatNode = {
    id,
    parentId,
    childrenIds: [],
    role,
    content,
    createdAt: Date.now(),
  };

  session.nodes[id] = node;

  if (parentId && session.nodes[parentId]) {
    session.nodes[parentId].childrenIds.push(id);
  } else {
    session.rootNodeId = id;
  }

  session.activeNodeId = id;

  if (session.title === "New Conversation" && role === "user") {
    session.title = content.slice(0, 32).trim() + (content.length > 32 ? "..." : "");
  }

  saveSession(session);
  return node;
}

export function editMessageNode(
  session: ChatSession,
  targetNodeId: string,
  newContent: string
): ChatNode {
  const oldNode = session.nodes[targetNodeId];
  if (!oldNode) return addMessageToSession(session, "user", newContent);

  const newNodeId = generateNodeId();
  const parentId = oldNode.parentId;

  const newNode: ChatNode = {
    id: newNodeId,
    parentId,
    childrenIds: [],
    role: oldNode.role,
    content: newContent,
    createdAt: Date.now(),
  };

  session.nodes[newNodeId] = newNode;

  if (parentId && session.nodes[parentId]) {
    session.nodes[parentId].childrenIds.push(newNodeId);
  } else {
    // Top-level root edit
    session.rootNodeId = newNodeId;
  }

  session.activeNodeId = newNodeId;
  saveSession(session);
  return newNode;
}

export function regenerateAssistantNode(session: ChatSession, assistantNodeId: string): ChatNode {
  const oldNode = session.nodes[assistantNodeId];
  const parentId = oldNode ? oldNode.parentId : session.activeNodeId;

  const newNodeId = generateNodeId();
  const newNode: ChatNode = {
    id: newNodeId,
    parentId,
    childrenIds: [],
    role: "assistant",
    content: "",
    createdAt: Date.now(),
  };

  session.nodes[newNodeId] = newNode;

  if (parentId && session.nodes[parentId]) {
    session.nodes[parentId].childrenIds.push(newNodeId);
  }

  session.activeNodeId = newNodeId;
  saveSession(session);
  return newNode;
}

export function switchSiblingBranch(session: ChatSession, targetSiblingId: string): void {
  const targetNode = session.nodes[targetSiblingId];
  if (!targetNode) return;

  // Find the deepest leaf node following the chosen target
  let current: ChatNode = targetNode;
  while (current.childrenIds && current.childrenIds.length > 0) {
    const lastChildId = current.childrenIds[current.childrenIds.length - 1];
    if (lastChildId && session.nodes[lastChildId]) {
      current = session.nodes[lastChildId]!;
    } else {
      break;
    }
  }

  session.activeNodeId = current.id;
  saveSession(session);
}

export function getActiveConversationPath(session: ChatSession): ChatNode[] {
  if (!session.activeNodeId || !session.nodes[session.activeNodeId]) {
    return [];
  }

  const path: ChatNode[] = [];
  let currId: string | null = session.activeNodeId;

  while (currId && session.nodes[currId]) {
    const node: ChatNode | undefined = session.nodes[currId];
    if (!node) break;
    path.unshift(node);
    currId = node.parentId;
  }

  return path;
}

export function getSiblingInfo(
  session: ChatSession,
  nodeId: string
): { index: number; total: number; siblings: ChatNode[] } {
  const node = session.nodes[nodeId];
  if (!node) return { index: 1, total: 1, siblings: [] };

  if (!node.parentId) {
    const rootSiblings = Object.values(session.nodes).filter((n) => !n.parentId);
    const index = rootSiblings.findIndex((s) => s.id === nodeId);
    return {
      index: index >= 0 ? index + 1 : 1,
      total: rootSiblings.length,
      siblings: rootSiblings,
    };
  }

  const parent = session.nodes[node.parentId];
  if (!parent || !parent.childrenIds) {
    return { index: 1, total: 1, siblings: [node] };
  }

  const siblings: ChatNode[] = parent.childrenIds
    .map((id) => session.nodes[id])
    .filter((n): n is ChatNode => Boolean(n));

  const index = siblings.findIndex((s) => s.id === nodeId);
  return {
    index: index >= 0 ? index + 1 : 1,
    total: siblings.length,
    siblings,
  };
}
