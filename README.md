# blender-hq-node-screenshot
A Blender add-on to export a clean, high-quality screenshot or HD diagram of any node tree

#
Node Tree HQ Screenshot is a node editor utility that allows users to export their node trees as clean, high-resolution images — directly from the N panel with a single click. It works with all node editors in Blender: Geometry Nodes, Shader Editor, Compositor, and more.

The add-on offers two distinct export modes. The first captures the node editor viewport exactly as displayed, hiding all UI panels for a clean output at native screen resolution. The second reconstructs the entire node tree as a fully custom GPU-rendered diagram at unlimited resolution, independent of screen size — ideal for documentation, portfolios, or sharing setups at any scale.

The diagram mode accurately represents node bodies, socket types with color coding, value fields, dropdowns, checkboxes, vector components, Float Curve graphs with their control points, reroute nodes, frame nodes with labels, and Bézier connection links. Output is saved as a PNG directly to the Desktop by default.

# Installation
Download the ZIP file.

Open Blender and go to **Edit** > **Preferences** > **Add-ons**.

Click **Install**, select the ZIP file, and click **Install Add-on**.

Enable the add-on by checking the corresponding box.

Access **Node Tree HQ Screenshot** in the **N menu** (sidebar) under the **Export tab**.

# How to Use

### Screenshot Mode
1. **Open a node editor** and set up your view — zoom and frame your nodes as desired.
2. Press **Ctrl+Space** to maximize the editor for the best possible resolution.
3. Click **Screenshot HQ** to export a clean PNG of exactly what is currently displayed.

### HD Diagram Mode
1. **Open a node editor** containing the node tree you want to export.
2. Adjust the **Resolution (px/unit)** slider to control the output size (higher = larger and more detailed image).
3. Enable **Show Values** to include default values on unconnected input sockets.
4. Click **Render HD Diagram** to generate and export the full reconstructed diagram as a PNG.

# Features

**Two Export Modes** – Screenshot captures the live viewport; HD Diagram renders a fully custom image at any resolution.

**Unlimited Output Resolution** – The diagram renderer is independent of screen size, supporting outputs up to 8192×8192 px.

**Full Node Body Reconstruction** – Accurately draws headers, output sockets, dropdowns, fields, checkboxes, vector components (X/Y/Z), and Boolean values.

**Float Curve Graph** – Renders the curve shape using control point interpolation, including the coordinate bar and interpolation icons below the graph.

**Color-Coded Sockets** – Geometry, Vector, Value, Boolean, Rotation and other socket types are rendered with their native Blender colors.

**Bézier Connection Links** – Links between nodes are drawn as smooth Bézier curves, color-matched to the source socket type.

**Frame Node Support** – Frame nodes are rendered with a dark background, border, and label, grouped behind the nodes they contain.

**Node Group Title** – The name of the active node tree is displayed centered at the top of the exported diagram.

**Reroute Node Support** – Reroute nodes are represented as colored dots consistent with their socket type.

**Sharpness Control** – In screenshot mode, temporarily boosts the UI scale before capture for crisper text and lines at the same resolution.

**Desktop Output by Default** – Exports are saved directly to the user's Desktop, with a customizable output folder in the panel.
