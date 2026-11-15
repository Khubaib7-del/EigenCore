// The one Node API vite.config.ts uses (the app itself has no Node types).
declare module "node:fs" {
  export function readFileSync(path: string | URL, encoding: "utf8"): string;
}
