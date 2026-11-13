import { useEffect } from "react";
import { CommandPalette, ResizeGrips, Sidebar, TitleBar, Toast } from "./components/Shell";
import { ChatScreen } from "./screens/ChatScreen";
import { ModelsScreen } from "./screens/ModelsScreen";
import { MonitorScreen } from "./screens/MonitorScreen";
import { ProfileScreen } from "./screens/ProfileScreen";
import { SettingsScreen } from "./screens/SettingsScreen";
import { resolvedTheme, useStore } from "./state/store";

export function App() {
  const { state } = useStore();
  const theme = resolvedTheme(state);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);
  const confirmDelete = new URLSearchParams(location.search).get("confirm") ?? undefined;
  return (
    <div className="app">
      <TitleBar />
      <div className="body">
        <Sidebar />
        <main className="main" key={state.screen}>
          {state.screen === "chat" && <ChatScreen />}
          {state.screen === "models" && <ModelsScreen confirmDelete={confirmDelete} />}
          {state.screen === "monitor" && <MonitorScreen />}
          {state.screen === "profile" && <ProfileScreen />}
          {state.screen === "settings" && <SettingsScreen />}
        </main>
      </div>
      <CommandPalette />
      <Toast />
      <ResizeGrips />
    </div>
  );
}
