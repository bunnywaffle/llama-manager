import { defineConfig } from "mygo-cli";

export default defineConfig({
  name: "llama-manager",
  identifier: "com.example.llamamanager",
  version: "0.1.1",
  icon: "resources/icon.png",
  // `mygo dev` runs devCommand and loads devUrl; `mygo build` runs
  // buildCommand and embeds frontendDist into the app.
  devUrl: "http://localhost:5173",
  devCommand: "npm run dev:web",
  buildCommand: "npm run build:web",
  frontendDist: "dist",
  bindings: "src/mygo.ts",
  out: "build",
});
