import { defineConfig, loadEnv } from "vite";
import path from "path";

export default defineConfig(({ mode }) => {
  // Load .env variables
  const env = loadEnv(mode, process.cwd(), "");

  return {
    base: mode === "development" ? "/" : env.VITE_BASE_PATH || "/",
    optimizeDeps: {
      entries: ["src/main.tsx"],
    },
    plugins: [],
    resolve: {
      preserveSymlinks: true,
      alias: {
        "@": path.resolve(__dirname, "./src"),
      },
    },
    server: {
      allowedHosts: true, // No need for @ts-ignore
    },
    assetsInclude: ["**/*.csv"],
  };
});
