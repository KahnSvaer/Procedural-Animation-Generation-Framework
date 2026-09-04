import "./index.css";

import Toolbar from "./editor/Toolbar/Toolbar";
import Viewport from "./editor/Viewport/Viewport";
import Inspector from "./editor/Inspector/Inspector";
import Timeline from "./editor/Timeline/Timeline";

import {
  Group,
  Panel,
  Separator,
} from "react-resizable-panels";

function App() {
  return (
    <div className="app">
      <Toolbar />

      <Group orientation="horizontal" className="workspace">
        <Panel defaultSize="70%">
          <Group orientation="vertical">
            <Panel defaultSize="70%">
              <Viewport />
            </Panel>
            <Separator className="resize-handle" />
            <Panel defaultSize="30%" minSize="140px">
              <Timeline />
            </Panel>
          </Group>
        </Panel>

        <Separator className="resize-handle" />

        <Panel
          defaultSize="30%"
          minSize="280px"
          maxSize="600px"
        >
          <Inspector />
        </Panel>
      </Group>
    </div>
  );
}

export default App;