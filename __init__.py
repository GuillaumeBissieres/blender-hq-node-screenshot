import bpy
import os
import time
import gpu
import blf
import numpy as np
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix

DEFAULT_DESKTOP = os.path.join(os.path.expanduser("~"), "Desktop")

# ---------------------------------------------------------------------------
# Layout constants (in node-space units, like Blender)
# ---------------------------------------------------------------------------
ROW_H       = 22
HEADER_H    = 26
PAD_X       = 10    # inner left/right padding
MARGIN      = 60    # marge autour du graphe
MAX_BUF     = 8192

# Couleurs
SOCKET_COLORS = {
    'GEOMETRY': (0.16, 0.60, 0.44, 1.0),
    'VALUE':    (0.63, 0.63, 0.63, 1.0),
    'INT':      (0.15, 0.55, 0.44, 1.0),
    'BOOLEAN':  (0.80, 0.28, 0.30, 1.0),
    'VECTOR':   (0.39, 0.39, 0.78, 1.0),
    'STRING':   (0.60, 0.40, 0.79, 1.0),
    'RGBA':     (0.78, 0.78, 0.16, 1.0),
    'ROTATION': (0.60, 0.40, 0.79, 1.0),
    'MATRIX':   (0.45, 0.30, 0.60, 1.0),
    'OBJECT':   (0.93, 0.60, 0.30, 1.0),
    'COLLECTION':(0.93, 0.60, 0.30, 1.0),
    'IMAGE':    (0.93, 0.60, 0.30, 1.0),
    'MATERIAL': (0.93, 0.60, 0.30, 1.0),
    'TEXTURE':  (0.93, 0.60, 0.30, 1.0),
}
DEFAULT_SOCKET_COLOR = (0.60, 0.60, 0.60, 1.0)
HEADER_DEFAULT       = (0.18, 0.18, 0.19, 1.0)
BODY_COLOR           = (0.13, 0.13, 0.14, 1.0)
BG_COLOR             = (0.086, 0.086, 0.086, 1.0)
FIELD_COLOR          = (0.22, 0.22, 0.24, 1.0)
DROPDOWN_COLOR       = (0.19, 0.19, 0.20, 1.0)
CHECKBOX_ON          = (0.35, 0.35, 0.37, 1.0)
TEXT_COLOR           = (0.92, 0.92, 0.92, 1.0)
DIM_TEXT_COLOR       = (0.60, 0.60, 0.60, 1.0)
VALUE_COLOR          = (0.75, 0.75, 0.75, 1.0)

AXIS_LABELS = ['X', 'Y', 'Z', 'W']

CATEGORY_COLORS = (
    (('GeometryNode', 'NodeGroupOutput', 'NodeGroupInput'),
     (0.145, 0.46, 0.39, 1.0)),
    (('FunctionNodeInput', 'ShaderNodeValue', 'GeometryNodeInput'),
     (0.45, 0.16, 0.16, 1.0)),
    (('ShaderNodeCombineXYZ', 'ShaderNodeSeparateXYZ', 'ShaderNodeVectorMath',
      'ShaderNodeMath', 'ShaderNodeFloat', 'ShaderNodeVector',
      'FunctionNodeRandom', 'FunctionNodeCompare', 'FunctionNodeBoolean',
      'FunctionNodeFloat', 'FunctionNodeRotate', 'FunctionNodeAlignEuler'),
     (0.165, 0.34, 0.52, 1.0)),
)

BASE_NODE_PROPS = {p.identifier for p in bpy.types.Node.bl_rna.properties}

# ---------------------------------------------------------------------------
# Helpers GPU
# ---------------------------------------------------------------------------
RECT_IDX = [(0,1,2),(0,2,3)]

def ortho_matrix(w, h):
    m = Matrix.Identity(4)
    m[0][0] = 2.0/w; m[1][1] = 2.0/h; m[2][2] = -1.0
    m[0][3] = -1.0;  m[1][3] = -1.0
    return m

def draw_rect(sh, x0, y0, x1, y1, col):
    b = batch_for_shader(sh,'TRIS',{"pos":[(x0,y0),(x1,y0),(x1,y1),(x0,y1)]},indices=RECT_IDX)
    sh.uniform_float("color",col); b.draw(sh)

def draw_rounded_rect(sh, x0, y0, x1, y1, col, r=None, seg=6):
    if r is None: r = min(4.0, (x1-x0)*0.08, (y1-y0)*0.3)
    r = max(0.0, min(r, (x1-x0)/2, (y1-y0)/2))
    verts = []
    def arc(cx, cy, a_start, a_end):
        for i in range(seg+1):
            a = a_start + (a_end-a_start)*i/seg
            verts.append((cx + r*np.cos(a), cy + r*np.sin(a)))
    arc(x1-r, y1-r,  0,          np.pi/2)
    arc(x0+r, y1-r,  np.pi/2,    np.pi)
    arc(x0+r, y0+r,  np.pi,      3*np.pi/2)
    arc(x1-r, y0+r,  3*np.pi/2,  2*np.pi)
    n = len(verts)
    cx, cy = (x0+x1)/2, (y0+y1)/2
    all_v = [(cx,cy)] + verts
    indices = [(0, i+1, (i+1)%n+1) for i in range(n)]
    b = batch_for_shader(sh,'TRIS',{"pos":all_v},indices=indices)
    sh.uniform_float("color",col); b.draw(sh)

def draw_circle(sh, cx, cy, r, col, seg=14):
    v = [(cx,cy)]
    for i in range(seg+1):
        a = i/seg*6.28318530718
        v.append((cx+r*np.cos(a), cy+r*np.sin(a)))
    b = batch_for_shader(sh,'TRIS',{"pos":v},indices=[(0,i,i+1) for i in range(1,seg+1)])
    sh.uniform_float("color",col); b.draw(sh)

def draw_line(sh, pts, col, w=1.5):
    gpu.state.line_width_set(w)
    b = batch_for_shader(sh,'LINE_STRIP',{"pos":pts})
    sh.uniform_float("color",col); b.draw(sh)
    gpu.state.line_width_set(1.0)

def draw_text(font_id, text, x, y):
    blf.position(font_id, x, y, 0)
    blf.draw(font_id, text)

def text_dim(font_id, text):
    return blf.dimensions(font_id, text)

# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------
def node_abs_location(node):
    x, y = node.location.x, node.location.y
    p = node.parent
    while p:
        x += p.location.x; y += p.location.y; p = p.parent
    return x, y

def node_header_color(node):
    if getattr(node,'use_custom_color',False):
        return tuple(node.color)+(1.0,)
    for prefixes, col in CATEGORY_COLORS:
        if node.bl_idname.startswith(prefixes):
            return col
    return HEADER_DEFAULT

def bezier_links(p0, p1, seg=20):
    dx = abs(p1[0]-p0[0])
    h = max(dx*0.5, 30)
    c0 = (p0[0]+h, p0[1]); c1 = (p1[0]-h, p1[1])
    pts = []
    for i in range(seg+1):
        t=i/seg; mt=1-t
        pts.append((
            mt**3*p0[0]+3*mt**2*t*c0[0]+3*mt*t**2*c1[0]+t**3*p1[0],
            mt**3*p0[1]+3*mt**2*t*c0[1]+3*mt*t**2*c1[1]+t**3*p1[1],
        ))
    return pts

def catmull_rom(ctrl, samples=8):
    n = len(ctrl)
    if n < 2: return ctrl
    pts = []
    for i in range(n-1):
        p0=ctrl[max(i-1,0)]; p1=ctrl[i]; p2=ctrl[i+1]; p3=ctrl[min(i+2,n-1)]
        for j in range(samples):
            t=j/samples; t2=t*t; t3=t2*t
            pts.append((
                0.5*((2*p1[0])+(-p0[0]+p2[0])*t+(2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2+(-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3),
                0.5*((2*p1[1])+(-p0[1]+p2[1])*t+(2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2+(-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3),
            ))
    pts.append(ctrl[-1])
    return pts

# ---------------------------------------------------------------------------
# Typed "body rows" system
# Each row = dict with 'kind' + data depending on type:
#   'out'      : output socket (label on right, dot on right)
#   'sep'      : empty separator
#   'dropdown' : menu pleine largeur  {'label': str}
#   'field'    : numeric field       {'label': str, 'value': str}
#   'checkbox' : checkbox             {'label': str, 'checked': bool}
#   'in_label' : input socket, no value  {'socket': s}
#   'in_field' : input socket with scalar value  {'socket': s, 'label': str, 'value': str}
#   'in_vec_hdr': vector input header  {'socket': s, 'label': str}
#   'in_vec_comp': composante X/Y/Z  {'axis': str, 'value': str}
#   'curve'    : zone courbe (hauteur variable)  {'node': n}
# ---------------------------------------------------------------------------
def build_body_rows(node, show_values):
    rows = []
    outs  = [s for s in node.outputs if not s.hide and s.enabled]
    ins   = [s for s in node.inputs  if not s.hide and s.enabled]
    has_curve = hasattr(node,'mapping') and getattr(node.mapping,'curves',None) is not None

    # --- Sorties ---
    for s in outs:
        rows.append({'kind':'out','socket':s,'label':s.name})
        # Nodes Value : valeur scalaire dans un champ
        if s.type == 'VALUE' and node.bl_idname in (
            'ShaderNodeValue','FunctionNodeInputFloat',
            'GeometryNodeInputNamedAttribute',
        ):
            try:
                val = node.outputs[0].default_value
                rows.append({'kind':'out_value','label':f"{val:.3f}"})
            except Exception:
                pass
        # Vector/Color nodes: X/Y/Z values split across multiple lines
        elif s.type in ('VECTOR','RGBA','ROTATION') and node.bl_idname in (
            'FunctionNodeInputVector','ShaderNodeRGB',
            'FunctionNodeInputColor','FunctionNodeInputRotation',
            'ShaderNodeCombineXYZ',
        ):
            try:
                vec = node.outputs[0].default_value
                for i, comp in enumerate(vec):
                    label = AXIS_LABELS[i] if i < len(AXIS_LABELS) else str(i)
                    rows.append({'kind':'out_vec_comp','axis':label,'value':f"{comp:.3f}"})
            except Exception:
                pass

    # --- Node-specific properties (not sockets) ---
    if not has_curve:
        for p in node.bl_rna.properties:
            ident = p.identifier
            if ident in BASE_NODE_PROPS or ident.startswith('bl_') or ident=='rna_type':
                continue
            if p.is_readonly:
                continue
            try:
                val = getattr(node, ident)
            except Exception:
                continue
            try:
                if p.type == 'ENUM':
                    rows.append({'kind':'dropdown','label':p.name,'value':str(val)})
                elif p.type == 'BOOLEAN':
                    rows.append({'kind':'checkbox','label':p.name,'checked':bool(val)})
                elif p.type == 'FLOAT' and hasattr(val,'__len__'):
                    pass  # vector property — rare, skip
                elif p.type in ('FLOAT','INT'):
                    rows.append({'kind':'field','label':p.name,'value':f"{val:.3f}" if p.type=='FLOAT' else str(val)})
            except Exception:
                continue

    # --- Courbe ---
    if has_curve:
        rows.append({'kind':'curve','node':node})
        # Coordinate bar (x/y values of active point) — just below graph
        try:
            crv = node.mapping.curves[0]
            pts = sorted(crv.points, key=lambda p: p.location[0])
            mid = min(pts, key=lambda p: abs(p.location[0]-0.5))
            rows.append({'kind':'curve_coords',
                         'x': f"{mid.location[0]:.4f}",
                         'y': f"{mid.location[1]:.4f}"})
        except Exception:
            pass

    # --- Inputs ---
    for s in ins:
        if s.is_linked or not show_values:
            rows.append({'kind':'in_label','socket':s,'label':s.name})
            continue
        try:
            raw = s.default_value
        except AttributeError:
            rows.append({'kind':'in_label','socket':s,'label':s.name})
            continue
        if hasattr(raw,'__len__') and not isinstance(raw,str) and len(raw)>1:
            rows.append({'kind':'in_vec_hdr','socket':s,'label':s.name})
            for i,comp in enumerate(raw):
                ax = AXIS_LABELS[i] if i<len(AXIS_LABELS) else str(i)
                try:   vstr = f"{comp:.3f}"
                except: vstr = str(comp)
                rows.append({'kind':'in_vec_comp','axis':ax,'value':vstr})
        else:
            try:
                if   isinstance(raw,float): vstr=f"{raw:.3f}"
                elif isinstance(raw,int):   vstr=str(raw)
                elif isinstance(raw,bool):  vstr="☑" if raw else "☐"
                elif isinstance(raw,str):   vstr=raw
                else:                       vstr=""
            except: vstr=""
            if vstr:
                rows.append({'kind':'in_field','socket':s,'label':s.name,'value':vstr})
            else:
                rows.append({'kind':'in_label','socket':s,'label':s.name})

    return rows, outs, ins

CURVE_H = 120   # estimated curve area height in node-space units

def rows_total_height(rows):
    h = HEADER_H
    for r in rows:
        if r['kind']=='curve': h += CURVE_H
        elif r['kind']=='curve_coords': h += ROW_H
        else: h += ROW_H
    return h

# ---------------------------------------------------------------------------
# Main operator: screenshot
# ---------------------------------------------------------------------------
def get_window_region(area):
    if area is None: return None
    for r in area.regions:
        if r.type=='WINDOW': return r

class NODE_OT_hq_screenshot(bpy.types.Operator):
    """Cache l'UI, cadre et exporte un PNG propre du node tree. Maximise
    the editor (Ctrl+Space) before clicking for best resolution."""
    bl_idname = "node.hq_screenshot"
    bl_label  = "Screenshot HQ du Node Tree"
    bl_options= {'REGISTER'}

    _timer=None; _state='INIT'; _ticks=0; TICKS=3

    def invoke(self,context,event):
        space=context.space_data
        if space is None or space.type!='NODE_EDITOR':
            self.report({'ERROR'},"Run this operator from a Node Editor"); return{'CANCELLED'}
        if space.edit_tree is None:
            self.report({'ERROR'},"No active node tree"); return{'CANCELLED'}
        region=get_window_region(context.area)
        if region is None:
            self.report({'ERROR'},"Editor region not found"); return{'CANCELLED'}
        scene=context.scene
        out_dir=bpy.path.abspath(scene.nhqs_output_dir)
        os.makedirs(out_dir,exist_ok=True)
        ts=time.strftime("%Y%m%d_%H%M%S")
        safe=bpy.path.clean_name(space.edit_tree.name)
        self.filepath=os.path.join(out_dir,f"{safe}_{ts}.png")
        self.window=context.window; self.area=context.area; self.region=region
        self.show_toolbar=space.show_region_toolbar
        self.show_ui=space.show_region_ui
        self.show_header=space.show_region_header
        self.orig_scale=context.preferences.view.ui_scale
        space.show_region_toolbar=False
        space.show_region_ui=False
        space.show_region_header=False
        context.preferences.view.ui_scale=min(self.orig_scale*scene.nhqs_quality_scale,3.0)
        region.tag_redraw()
        self._state='WAIT_HIDE'; self._ticks=0
        wm=context.window_manager
        self._timer=wm.event_timer_add(0.05,window=context.window)
        wm.modal_handler_add(self); return{'RUNNING_MODAL'}

    def modal(self,context,event):
        if event.type!='TIMER': return{'PASS_THROUGH'}
        try:
            self._ticks+=1
            if self._state=='WAIT_HIDE':
                if self._ticks>=self.TICKS:
                    with context.temp_override(window=self.window,area=self.area,region=self.region):
                        bpy.ops.screen.screenshot_area(filepath=self.filepath)
                    self.report({'INFO'},f"Screenshot saved: {self.filepath}")
                    self._state='DONE'
                return{'RUNNING_MODAL'}
            self._finish(context); return{'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'},f"Erreur : {exc}"); self._finish(context); return{'CANCELLED'}

    def _finish(self,context):
        if self._timer:
            context.window_manager.event_timer_remove(self._timer); self._timer=None
        context.preferences.view.ui_scale=self.orig_scale
        sp=self.area.spaces.active if self.area else None
        if sp:
            sp.show_region_toolbar=self.show_toolbar
            sp.show_region_ui=self.show_ui
            sp.show_region_header=self.show_header
        if self.region: self.region.tag_redraw()

    def cancel(self,context): self._finish(context)

# ---------------------------------------------------------------------------
# HD diagram render operator
# ---------------------------------------------------------------------------
class NODE_OT_hq_diagram_render(bpy.types.Operator):
    """Renders the node tree as an offscreen GPU diagram at unlimited
    resolution and exports a clean PNG to the Desktop."""
    bl_idname = "node.hq_diagram_render"
    bl_label  = "Render HD Diagram"
    bl_options= {'REGISTER'}

    def execute(self,context):
        space=context.space_data
        if space is None or space.type!='NODE_EDITOR':
            self.report({'ERROR'},"Run this operator from a Node Editor"); return{'CANCELLED'}
        node_tree=space.edit_tree
        if node_tree is None:
            self.report({'ERROR'},"No active node tree"); return{'CANCELLED'}
        scene=context.scene
        show_values=scene.nhqs_show_values
        frames=[n for n in node_tree.nodes if n.type=='FRAME']
        nodes=[n for n in node_tree.nodes if n.type not in ('FRAME',)]
        if not nodes:
            self.report({'ERROR'},"No nodes to export"); return{'CANCELLED'}
        group_name = node_tree.name

        # --- Construire les rows et calculer les tailles ---
        node_data = {}
        min_x=min_y=float('inf'); max_x=max_y=float('-inf')

        # Include frames in the bounding box
        frame_data = []
        for f in frames:
            ax,ay = node_abs_location(f)
            w = max(f.width, 40)
            h = max(f.height, 40)
            frame_data.append((f, ax, ay, w, h))
            min_x=min(min_x,ax); max_x=max(max_x,ax+w)
            min_y=min(min_y,ay-h); max_y=max(max_y,ay)

        for n in nodes:
            ax,ay = node_abs_location(n)
            is_reroute = (n.type=='REROUTE')
            if is_reroute:
                w,h=12,12; rows=[]; outs=[]; ins=[]
            else:
                rows,outs,ins = build_body_rows(n, show_values)
                w = max(n.width, 40)
                h = rows_total_height(rows)
            node_data[n] = (ax,ay,w,h,rows,outs,ins,is_reroute)
            min_x=min(min_x,ax); max_x=max(max_x,ax+w)
            min_y=min(min_y,ay-h); max_y=max(max_y,ay)

        scale=scene.nhqs_diagram_scale
        cw=(max_x-min_x+2*MARGIN)*scale
        ch=(max_y-min_y+2*MARGIN)*scale
        if max(cw,ch)>MAX_BUF:
            f=MAX_BUF/max(cw,ch); scale*=f; cw*=f; ch*=f
        W,H=int(cw),int(ch)

        def tb(x,y):
            return ((x-min_x+MARGIN)*scale, (y-min_y+MARGIN)*scale)

        # --- Positions des sockets ---
        socket_pos={}
        for n,(_,__,___,____,rows,outs,ins,is_reroute) in node_data.items():
            ax,ay,w,h=node_data[n][:4]
            if is_reroute:
                cx,cy=tb(ax+6,ay-6)
                for sl in [*n.inputs,*n.outputs]: socket_pos[sl]=(cx,cy)
                continue
            # Current Y position from top
            y_cur = ay - HEADER_H
            for r in rows:
                rh = CURVE_H if r['kind']=='curve' else ROW_H
                row_cy = y_cur - rh/2
                if r['kind']=='out':
                    socket_pos[r['socket']] = tb(ax+w, row_cy)
                elif r['kind'] in ('in_label','in_field','in_vec_hdr'):
                    socket_pos[r['socket']] = tb(ax, row_cy)
                y_cur -= rh

        # --- Render ---
        try:
            offscreen=gpu.types.GPUOffScreen(W,H)
        except Exception as exc:
            self.report({'ERROR'},f"Failed to create GPU buffer: {exc}"); return{'CANCELLED'}

        sh=gpu.shader.from_builtin('UNIFORM_COLOR')
        proj=ortho_matrix(W,H)

        with offscreen.bind():
            fb=gpu.state.active_framebuffer_get()
            fb.clear(color=BG_COLOR)
            gpu.matrix.push(); gpu.matrix.push_projection()
            gpu.matrix.load_matrix(Matrix.Identity(4))
            gpu.matrix.load_projection_matrix(proj)
            gpu.state.blend_set('ALPHA')

            dot_r=max(4.0, scale*2.2)
            link_w=max(2.5, scale*1.0)

            # --- Frames (drawn first, behind everything) ---
            FRAME_BG    = (0.06, 0.06, 0.07, 0.92)
            FRAME_BORDER= (0.25, 0.25, 0.27, 1.0)
            FRAME_HDR   = (0.10, 0.10, 0.12, 1.0)
            for f, ax, ay, w, h in frame_data:
                x0,y0=tb(ax,ay-h); x1,y1=tb(ax+w,ay)
                # Bordure fine
                bw = max(1.5, scale*0.5)
                draw_rounded_rect(sh,x0-bw,y0-bw,x1+bw,y1+bw,FRAME_BORDER,r=6*scale)
                # Very dark background
                draw_rounded_rect(sh,x0,y0,x1,y1,FRAME_BG,r=6*scale)
                # Title header
                hx0,hy0=tb(ax,ay-HEADER_H); hx1,hy1=tb(ax+w,ay)
                draw_rounded_rect(sh,hx0,hy0,hx1,hy1,FRAME_HDR,r=6*scale)

            # --- Liens ---
            ERROR_LINK_COLOR = (0.80, 0.08, 0.08, 0.95)
            for link in node_tree.links:
                p0=socket_pos.get(link.from_socket)
                p1=socket_pos.get(link.to_socket)
                if p0 and p1:
                    is_err = not link.is_valid or link.is_muted
                    col = ERROR_LINK_COLOR if is_err else SOCKET_COLORS.get(link.from_socket.type,DEFAULT_SOCKET_COLOR)
                    pts = bezier_links(p0,p1)
                    draw_line(sh,pts,(col[0],col[1],col[2],0.9),link_w)
                    # Warning triangle on the link itself (at midpoint)
                    if is_err and len(pts) >= 2:
                        mid = pts[len(pts)//2]
                        ts = max(10.0, scale * 8.0)
                        tx, ty = mid[0] - ts*0.5, mid[1]
                        tri = [(tx, ty - ts*0.55), (tx + ts, ty - ts*0.55), (tx + ts*0.5, ty + ts*0.55)]
                        tb_ = batch_for_shader(sh,'TRIS',{"pos":tri},indices=[(0,1,2)])
                        sh.uniform_float("color",(0.95,0.55,0.05,1.0)); tb_.draw(sh)
                        # Exclamation mark
                        ex = tx + ts*0.5 - ts*0.07
                        draw_rect(sh, ex, ty-ts*0.28, ex+ts*0.14, ty+ts*0.22, (0.10,0.07,0.02,1.0))

            # --- Nodes ---
            for n in nodes:
                ax,ay,w,h,rows,outs,ins,is_reroute=node_data[n]
                if is_reroute:
                    cx,cy=tb(ax+6,ay-6)
                    sock=(n.outputs or n.inputs or [None])[0]
                    col=SOCKET_COLORS.get(sock.type,DEFAULT_SOCKET_COLOR) if sock else DEFAULT_SOCKET_COLOR
                    draw_circle(sh,cx,cy,dot_r*1.4,col); continue

                x0,y0=tb(ax,ay-h); x1,y1=tb(ax+w,ay)
                draw_rounded_rect(sh,x0,y0,x1,y1,BODY_COLOR,r=5*scale)
                hx0,hy0=tb(ax,ay-HEADER_H); hx1,hy1=tb(ax+w,ay)
                draw_rounded_rect(sh,hx0,hy0,hx1,hy1,node_header_color(n),r=5*scale)

                # Warning triangle for nodes with errors
                if getattr(n, 'use_custom_color', False) is False:
                    pass

                y_cur=ay-HEADER_H
                for r in rows:
                    rh=CURVE_H if r['kind']=='curve' else ROW_H
                    row_cy=y_cur-rh/2
                    bx0,by0=tb(ax+PAD_X,y_cur-rh+3)
                    bx1,by1=tb(ax+w-PAD_X,y_cur-3)

                    if r['kind']=='out':
                        p=socket_pos[r['socket']]
                        draw_circle(sh,p[0],p[1],dot_r,SOCKET_COLORS.get(r['socket'].type,DEFAULT_SOCKET_COLOR))

                    elif r['kind']=='out_value':
                        draw_rect(sh,bx0,by0,bx1,by1,FIELD_COLOR)

                    elif r['kind']=='out_vec_comp':
                        draw_rect(sh,bx0,by0,bx1,by1,FIELD_COLOR)

                    elif r['kind']=='curve_coords':
                        draw_rect(sh,bx0,by0,bx1,by1,DROPDOWN_COLOR)

                    elif r['kind']=='dropdown':
                        draw_rect(sh,bx0,by0,bx1,by1,DROPDOWN_COLOR)

                    elif r['kind']=='field':
                        draw_rect(sh,bx0,by0,bx1,by1,FIELD_COLOR)

                    elif r['kind']=='checkbox':
                        cb_size=max(6,ROW_H*0.5)*scale
                        cbx0,cby0=tb(ax+PAD_X,row_cy-ROW_H*0.28)
                        cbx1=cbx0+cb_size; cby1=cby0+cb_size
                        draw_rect(sh,cbx0,cby0,cbx1,cby1,CHECKBOX_ON if r['checked'] else DROPDOWN_COLOR)

                    elif r['kind']=='in_label':
                        p=socket_pos[r['socket']]
                        draw_circle(sh,p[0],p[1],dot_r,SOCKET_COLORS.get(r['socket'].type,DEFAULT_SOCKET_COLOR))

                    elif r['kind']=='in_field':
                        p=socket_pos[r['socket']]
                        draw_circle(sh,p[0],p[1],dot_r,SOCKET_COLORS.get(r['socket'].type,DEFAULT_SOCKET_COLOR))
                        draw_rect(sh,bx0,by0,bx1,by1,FIELD_COLOR)

                    elif r['kind']=='in_vec_hdr':
                        p=socket_pos[r['socket']]
                        draw_circle(sh,p[0],p[1],dot_r,SOCKET_COLORS.get(r['socket'].type,DEFAULT_SOCKET_COLOR))

                    elif r['kind']=='in_vec_comp':
                        draw_rect(sh,bx0,by0,bx1,by1,FIELD_COLOR)

                    elif r['kind']=='curve':
                        try:
                            mapping = n.mapping
                            crv = mapping.curves[0]
                            ICON_BAR_H = ROW_H * 1.2  # icon+coords bar height
                            czx0,czy0=tb(ax+PAD_X, ay-h+PAD_X + ROW_H*2 + ICON_BAR_H)
                            czx1,czy1=tb(ax+w-PAD_X, y_cur)
                            if czx1>czx0 and czy1>czy0:
                                draw_rect(sh,czx0,czy0,czx1,czy1,(0.06,0.06,0.07,1.0))
                                ctrl=sorted(
                                    ((p.location[0],p.location[1]) for p in crv.points),
                                    key=lambda t:t[0]
                                )
                                if len(ctrl)>=2:
                                    mapped=[]
                                    for cx_,cy_ in ctrl:
                                        t=max(0.0,min(1.0,cx_)); v=max(0.0,min(1.0,cy_))
                                        mapped.append((czx0+t*(czx1-czx0), czy0+v*(czy1-czy0)))
                                    smooth=catmull_rom(mapped, samples=16)
                                    draw_line(sh,smooth,(0.9,0.6,0.2,1.0),max(2.0,scale*0.9))
                                    for px,py in mapped:
                                        draw_circle(sh,px,py,max(2.0,scale*1.2),(0.95,0.95,0.95,1.0))
                                # Icon + coordinate bar below the graph
                                ibar_y1 = czy0
                                ibar_y0 = ibar_y1 - ICON_BAR_H * scale
                                draw_rect(sh,czx0,ibar_y0,czx1,ibar_y1,(0.14,0.14,0.16,1.0))
                                # 3 interpolation icons (Auto, Vector, Auto Clamped)
                                # drawn as symbolic mini-curves
                                icon_s = ICON_BAR_H * scale * 0.55
                                icon_pad = icon_s * 0.3
                                for i in range(3):
                                    ix0 = czx0 + icon_pad + i*(icon_s + icon_pad)
                                    iy_mid = (ibar_y0 + ibar_y1) / 2
                                    ix1 = ix0 + icon_s
                                    # icon background
                                    draw_rect(sh,ix0-2,ibar_y0+2,ix1+2,ibar_y1-2,(0.22,0.22,0.26,1.0))
                                    # mini-courbe symbolique
                                    if i == 0:  # Auto : cloche
                                        pts_icon=[]
                                        for j in range(9):
                                            t=j/8; v=4*t*(1-t)
                                            pts_icon.append((ix0+t*icon_s, ibar_y0+4+(ibar_y1-ibar_y0-8)*v))
                                        draw_line(sh,pts_icon,(0.85,0.85,0.85,1.0),max(1.0,scale*0.4))
                                    elif i == 1:  # Vector : V
                                        mid_x=(ix0+ix1)/2
                                        draw_line(sh,[(ix0,ibar_y1-4),(mid_x,ibar_y0+4),(ix1,ibar_y1-4)],(0.85,0.85,0.85,1.0),max(1.0,scale*0.4))
                                    else:  # Auto Clamped : S
                                        pts_icon=[]
                                        for j in range(9):
                                            t=j/8; v=3*t**2-2*t**3
                                            pts_icon.append((ix0+t*icon_s, ibar_y0+4+(ibar_y1-ibar_y0-8)*v))
                                        draw_line(sh,pts_icon,(0.85,0.85,0.85,1.0),max(1.0,scale*0.4))
                        except Exception:
                            pass

                    elif r['kind']=='curve_coords':
                        draw_rect(sh,bx0,by0,bx1,by1,DROPDOWN_COLOR)

                    y_cur-=rh

            # --- Texte ---
            fid=0
            blf.size(fid,max(8,int(12*scale)))
            blf.color(fid,*TEXT_COLOR)

            # --- Node group title centered at top ---
            title_size = max(10, int(16*scale))
            blf.size(fid, title_size)
            blf.color(fid, 0.75, 0.75, 0.75, 1.0)
            tw,th = text_dim(fid, group_name)
            blf.position(fid, W/2 - tw/2, H - MARGIN*scale*0.5, 0)
            blf.draw(fid, group_name)
            blf.size(fid, max(8,int(12*scale)))

            # --- Labels des frames ---
            blf.color(fid, 0.80, 0.80, 0.80, 1.0)
            for f, ax, ay, w, h in frame_data:
                label = f.label if f.label else f.name
                hx0,hy0 = tb(ax, ay-HEADER_H)
                blf.position(fid, hx0+6*scale, hy0+6*scale, 0)
                blf.draw(fid, label)

            blf.color(fid,*TEXT_COLOR)
            for n in nodes:
                ax,ay,w,h,rows,outs,ins,is_reroute=node_data[n]
                if is_reroute: continue
                hx0,hy0=tb(ax,ay-HEADER_H)
                label=n.label if n.label else n.name
                blf.position(fid,hx0+5*scale,hy0+6*scale,0)
                blf.draw(fid,label)

                y_cur=ay-HEADER_H
                for r in rows:
                    rh=CURVE_H if r['kind']=='curve' else ROW_H
                    row_cy=y_cur-rh/2
                    _,th=text_dim(fid,"X")
                    ty=tb(ax,row_cy)[1]-th*0.4

                    if r['kind']=='out':
                        tw,_=text_dim(fid,r['label'])
                        blf.color(fid,*TEXT_COLOR)
                        blf.position(fid,tb(ax+w-PAD_X,0)[0]-tw,ty,0)
                        blf.draw(fid,r['label'])

                    elif r['kind']=='out_value':
                        tw,_=text_dim(fid,r['label'])
                        blf.color(fid,*VALUE_COLOR)
                        blf.position(fid,tb(ax+w-PAD_X*2,0)[0]-tw,ty,0)
                        blf.draw(fid,r['label'])
                        blf.color(fid,*TEXT_COLOR)

                    elif r['kind']=='out_vec_comp':
                        blf.color(fid,*DIM_TEXT_COLOR)
                        blf.position(fid,tb(ax+PAD_X*3,0)[0],ty,0)
                        blf.draw(fid,r['axis'])
                        tw,_=text_dim(fid,r['value'])
                        blf.color(fid,*VALUE_COLOR)
                        blf.position(fid,tb(ax+w-PAD_X*2,0)[0]-tw,ty,0)
                        blf.draw(fid,r['value'])
                        blf.color(fid,*TEXT_COLOR)

                    elif r['kind']=='curve_coords':
                        txt=f"{r['x']}   {r['y']}"
                        tw,_=text_dim(fid,txt)
                        blf.color(fid,*VALUE_COLOR)
                        blf.position(fid,tb(ax+w/2,0)[0]-tw/2,ty,0)
                        blf.draw(fid,txt)
                        blf.color(fid,*TEXT_COLOR)

                    elif r['kind']=='dropdown':
                        blf.color(fid,*TEXT_COLOR)
                        blf.position(fid,tb(ax+PAD_X*2,0)[0],ty,0)
                        blf.draw(fid,r['value'])

                    elif r['kind']=='field':
                        blf.color(fid,*DIM_TEXT_COLOR)
                        blf.position(fid,tb(ax+PAD_X*2,0)[0],ty,0)
                        blf.draw(fid,r['label'])
                        tw,_=text_dim(fid,r['value'])
                        blf.color(fid,*VALUE_COLOR)
                        blf.position(fid,tb(ax+w-PAD_X,0)[0]-tw,ty,0)
                        blf.draw(fid,r['value'])

                    elif r['kind']=='checkbox':
                        cb_size=max(6,ROW_H*0.5)*scale
                        blf.color(fid,*TEXT_COLOR)
                        blf.position(fid,tb(ax+PAD_X,0)[0]+cb_size+4*scale,ty,0)
                        blf.draw(fid,r['label'])

                    elif r['kind'] in ('in_label','in_vec_hdr'):
                        blf.color(fid,*TEXT_COLOR)
                        blf.position(fid,tb(ax+PAD_X,0)[0],ty,0)
                        blf.draw(fid,r['label'])

                    elif r['kind']=='in_field':
                        blf.color(fid,*DIM_TEXT_COLOR)
                        blf.position(fid,tb(ax+PAD_X*2,0)[0],ty,0)
                        blf.draw(fid,r['label'])
                        tw,_=text_dim(fid,r['value'])
                        blf.color(fid,*VALUE_COLOR)
                        blf.position(fid,tb(ax+w-PAD_X,0)[0]-tw,ty,0)
                        blf.draw(fid,r['value'])

                    elif r['kind']=='in_vec_comp':
                        blf.color(fid,*DIM_TEXT_COLOR)
                        blf.position(fid,tb(ax+PAD_X*3,0)[0],ty,0)
                        blf.draw(fid,r['axis'])
                        tw,_=text_dim(fid,r['value'])
                        blf.color(fid,*VALUE_COLOR)
                        blf.position(fid,tb(ax+w-PAD_X,0)[0]-tw,ty,0)
                        blf.draw(fid,r['value'])

                    y_cur-=rh

            gpu.matrix.pop_projection(); gpu.matrix.pop()
            buffer=fb.read_color(0,0,W,H,4,0,'FLOAT')

        offscreen.free()
        buffer.dimensions=W*H*4
        pixels=np.array(buffer,dtype=np.float32)

        out_dir=bpy.path.abspath(scene.nhqs_output_dir)
        os.makedirs(out_dir,exist_ok=True)
        ts=time.strftime("%Y%m%d_%H%M%S")
        safe=bpy.path.clean_name(node_tree.name)
        filepath=os.path.join(out_dir,f"{safe}_diagram_{ts}.png")

        img=bpy.data.images.new(f"nhqs_{ts}",W,H,alpha=True)
        img.pixels.foreach_set(pixels.ravel())
        img.filepath_raw=filepath; img.file_format='PNG'
        img.save(); bpy.data.images.remove(img)
        self.report({'INFO'},f"Diagram exported ({W}x{H}): {filepath}")
        return{'FINISHED'}

# ---------------------------------------------------------------------------
# Panneau
# ---------------------------------------------------------------------------
class NODE_PT_hq_screenshot_panel(bpy.types.Panel):
    bl_label       = "Screenshot HQ"
    bl_idname      = "NODE_PT_hq_screenshot_panel"
    bl_space_type  = 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_category    = "Export"

    def draw(self,context):
        layout=self.layout; scene=context.scene
        layout.prop(scene,"nhqs_output_dir")
        layout.prop(scene,"nhqs_quality_scale")
        layout.operator("node.hq_screenshot",icon='IMAGE_DATA')
        col=layout.column(); col.scale_y=0.8
        col.label(text="Frame/zoom as desired before clicking.",icon='INFO')
        col.label(text="Ctrl+Space to maximize the editor.")
        layout.separator()
        layout.label(text="Render HD Diagram")
        layout.prop(scene,"nhqs_diagram_scale")
        layout.prop(scene,"nhqs_show_values")
        layout.operator("node.hq_diagram_render",icon='RENDER_STILL')

# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
classes=(NODE_OT_hq_screenshot, NODE_OT_hq_diagram_render, NODE_PT_hq_screenshot_panel)

def register():
    for c in classes: bpy.utils.register_class(c)
    bpy.types.Scene.nhqs_output_dir=bpy.props.StringProperty(
        name="Output Folder",subtype='DIR_PATH',default=DEFAULT_DESKTOP)
    bpy.types.Scene.nhqs_quality_scale=bpy.props.FloatProperty(
        name="Sharpness (UI scale)",default=1.6,min=1.0,max=3.0)
    bpy.types.Scene.nhqs_diagram_scale=bpy.props.FloatProperty(
        name="Resolution (px/unit)",default=3.0,min=1.0,max=8.0)
    bpy.types.Scene.nhqs_show_values=bpy.props.BoolProperty(
        name="Show Values",default=True)

def unregister():
    for attr in ('nhqs_output_dir','nhqs_quality_scale','nhqs_diagram_scale','nhqs_show_values'):
        if hasattr(bpy.types.Scene,attr): delattr(bpy.types.Scene,attr)
    for c in reversed(classes): bpy.utils.unregister_class(c)
