import open3d as o3d
import numpy as np


# From : https://stackoverflow.com/a/59026582/8574085

def calculate_zy_rotation_for_arrow(v):
    """
    Calculates the rotations required to go from the vector v to the 
    z axis vector. The first rotation that is 
    calculated is over the z axis. This will leave the vector v on the
    XZ plane. Then, the rotation over the y axis. 

    Returns the angles of rotation over axis z and y required to
    get the vector v into the same orientation as axis z

    Args:
        - v (): 
    """
    # Rotation over z axis 
    gamma = np.arctan(v[1]/v[0])
    Rz = np.array([[np.cos(gamma),-np.sin(gamma),0],
                   [np.sin(gamma),np.cos(gamma),0],
                   [0,0,1]])
    # Rotate v to calculate next rotation
    v = Rz.T@v.reshape(-1,1)
    v = v.reshape(-1)
    # Rotation over y axis
    beta = np.arctan(v[0]/v[2])
    Ry = np.array([[np.cos(beta),0,np.sin(beta)],
                   [0,1,0],
                   [-np.sin(beta),0,np.cos(beta)]])
    return Rz @ Ry

def create_cylinder(height=1, radius=None, resolution=20):
    """
    Create an cylinder in Open3D
    """
    radius = height/20 if radius is None else radius
    mesh_frame = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius,
        height=height,
        resolution=resolution)
    return(mesh_frame)

def create_segment(a, b, radius=0.05, color=(1,1,0), resolution=20):
    """
    Creates an line(cylinder) from an pointa to point b,
    or create an line from a vector v starting from origin.
    Args:
        - a, b: End points [x,y,z]
        - radius: radius cylinder
    """
    a = np.array(a)
    b = np.array(b)
    T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T[:3, -1] = a
    v = b-a 

    height = np.linalg.norm(v)
    if height == 0: return None
    R = calculate_zy_rotation_for_arrow(v)
    mesh = create_cylinder(height, radius)
    mesh.rotate(R, center=np.array([0, 0, 0]))
    mesh.translate((a+b)/2)
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh

def create_tetra(p1, p2, p3, p4, color=(1,1,0)):
    vertices = o3d.utility.Vector3dVector([p1, p2, p3, p4])
    tetras = o3d.utility.Vector4iVector([[0, 1, 2, 3]])
    mesh = o3d.geometry.TetraMesh(vertices, tetras)
    mesh.paint_uniform_color(color)
    return mesh

def create_grid(p0, p1, p2, p3, ni1, ni2, color=(0,0,0)):
    '''
    p0, p1, p2, p3 : points defining a quadrilateral
    ni1: nb of equidistant intervals on segments p0p1 and p3p2
    ni2: nb of equidistant intervals on segments p1p2 and p0p3
    '''
    p0 = np.array(p0)
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    
    vertices = [p0, p1, p2, p3]
    lines = [[0,1],[0,3],[1,2],[2,3]]
    for i in range(1,ni1):
        l = len(vertices)
        vertices.append((p0*(ni1-i)+p1*i)/ni1)
        vertices.append((p3*(ni1-i)+p2*i)/ni1)
        lines.append([l,l+1])
    for i in range(1,ni2):
        l = len(vertices)
        vertices.append((p1*(ni2-i)+p2*i)/ni2)
        vertices.append((p0*(ni2-i)+p3*i)/ni2)
        lines.append([l,l+1])
    vertices = o3d.utility.Vector3dVector(vertices)
    lines = o3d.utility.Vector2iVector(lines)
    mesh = o3d.geometry.LineSet(vertices, lines)
    mesh.paint_uniform_color(color)
    return mesh


def create_coord_frame(origin=[0, 0, 0],size=1):
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    mesh.translate(origin)
    return mesh


# Visu3d : custom class used to visualize 3d skeleton
class Visu3D:
    def __init__(self, bg_color=[0,0,0], zoom=1, segment_radius=1):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window() 
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray(bg_color)
        # Defining callbacks - Key codes: https://www.glfw.org/docs/latest/group__keys.html
        self.vis.register_key_callback(ord("R"), self.start_rotating)
        self.vis.register_key_callback(ord("O"), self.start_oscillating)
        self.vis.register_key_callback(ord("S"), self.stop_moving)
        self.vis.register_key_callback(262, self.turn_view_right) # Right arrow
        self.vis.register_key_callback(263, self.turn_view_left)  # Left arrow
        self.vis.register_key_callback(265, self.incr_rot_speed) # Up arrow
        self.vis.register_key_callback(264, self.decr_rot_speed) # Down arrow
        self.view_control = self.vis.get_view_control()
        self.zoom = zoom
        self.segment_radius = segment_radius
        self.move = "oscillate"
        self.angle = 0
        self.direction = 1
        self.oscillate_angle = 200       
        self.geometries = []

    def set_view(self):
        if self.angle_view % 4 == 0:
            ax = 0
        elif self.angle_view <= 3:
            ax = 1
        else:
            ax = -1
        if self.angle_view == 2 or self.angle_view == 6:
            az = 0
        elif 3 <= self.angle_view <= 5:
            az = 1
        else:
            az = -1
        self.view_control.set_front(np.array([ax,0,az]))
        self.view_control.set_up(np.array([0,-1,0]))

    def init_view(self):
        self.angle_view = 0
        self.rot_speed = 2
        self.set_view()
        self.view_control.set_zoom(self.zoom)
        
    def create_grid(self, p0, p1, p2, p3, ni1, ni2, color=(1,1,1)):
        '''
        p0, p1, p2, p3 : points defining a quadrilateral
        ni1: nb of equidistant intervals on segments p0p1 and p3p2
        ni2: nb of equidistant intervals on segments p1p2 and p0p3
        '''
        grid = create_grid(p0, p1, p2, p3, ni1, ni2, color)
        self.vis.add_geometry(grid)
        self.geometries.append(grid)

    def create_camera(self):
        cam = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.02, cone_radius=0.03, cylinder_height=0.1, cone_height=0.08)
        cam.paint_uniform_color([0.2,0.7,1])
        cam.compute_vertex_normals()
        self.geometries.append(cam)

    def add_geometries(self):
        for geo in self.geometries:
            self.vis.add_geometry(geo, reset_bounding_box=False)

    def add_segment(self, p1, p2, radius=None, color=[1,1,1]):
        radius = self.segment_radius if radius is None else radius
        line = create_segment(p1, p2, radius=radius, color=color)
        if line: self.vis.add_geometry(line, reset_bounding_box=False)

    def clear(self):
        self.vis.clear_geometries()

    # Callback
    def incr_rot_speed(self, vis):
        if self.move == "rotate":
            if self.rot_speed * self.direction == -1:
                self.direction = 1
            else:
                self.rot_speed += self.direction
        else:
            self.rot_speed += 1
    # Callback
    def decr_rot_speed(self, vis):
        if self.move == "rotate":
            if self.rot_speed * self.direction == 1:
                self.direction = -1
            else:
                self.rot_speed -= self.direction
        else:
            self.rot_speed = max (1, self.rot_speed-1)
    # Callback
    def turn_view_right(self, vis):
        self.angle_view = (self.angle_view + 1) %8
        self.set_view()
        self.move = None
    # Callback
    def turn_view_left(self, vis):
        self.angle_view = (self.angle_view - 1) %8
        self.set_view()
        self.move = None
    # Callback
    def start_rotating(self, vis):
        self.move = "rotate"
    # Callback
    def start_oscillating(self, vis):
        self.move = "oscillate"
        self.angle = 0
    # Callback
    def stop_moving(self, vis):
        self.move = None

    def try_move(self):
        if self.move == "rotate":
            self.view_control.rotate(self.rot_speed * self.direction,0)
        elif self.move == "oscillate":
            self.view_control.rotate(self.rot_speed * self.direction,0)
            self.angle += self.rot_speed * self.direction
            if abs(self.angle) >= self.oscillate_angle:
                self.direction = - self.direction
            
    def render(self):
        self.vis.poll_events()
        self.vis.update_renderer()



if __name__ == "__main__":

    line = create_segment([0, 0, 0], [1, 0, 0],  color=(1,0,0))
    line2 = create_segment([1, 0, 0], [1, 1, 0], color=(0,1,0))
    line3 = create_segment([1, 1, 0], [0, 0, 0], radius=0.1)
    grid = create_grid([0,0,0],[0,0,1],[0,1,1],[0,1,0], 3, 2)
    frame =create_coord_frame()
    print(grid)
    # Draw everything
    o3d.visualization.draw_geometries([line, line2, line3, grid, frame])