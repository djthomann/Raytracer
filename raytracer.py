"""
/*******************************************************************************
 *
 *            #, #,         CCCCCC  VV    VV MM      MM RRRRRRR
 *           %  %(  #%%#   CC    CC VV    VV MMM    MMM RR    RR
 *           %    %## #    CC        V    V  MM M  M MM RR    RR
 *            ,%      %    CC        VV  VV  MM  MM  MM RRRRRR
 *            (%      %,   CC    CC   VVVV   MM      MM RR   RR
 *              #%    %*    CCCCCC     VV    MM      MM RR    RR
 *             .%    %/
 *                (%.      Computer Vision & Mixed Reality Group
 *
 ******************************************************************************/
/**          @copyright:   Hochschule RheinMain,
 *                         University of Applied Sciences
 *              @author:   Prof. Dr. Ulrich Schwanecke, Fabian Stahl
 *             @version:   2.0
 *                @date:   01.04.2023
 ******************************************************************************/
/**         raytracerTemplate.py
 *
 *          Simple Python template to generate ray traced images and display
 *          results in a 2D scene using OpenGL.
 ****
"""

from PIL import Image
from rendering import Scene, RenderWindow
import numpy as np
import numpy.linalg as lg
from rt3 import vec3, Sphere, Plane, Triangle, FARAWAY, L, E, rgb
import rt3 as rt3
from functools import reduce
import numbers

class RayTracer:

    def __init__(self, width, height):
        self.width  = width
        self.height = height

        # Scene info
        self.e = E
        self.c = vec3(1, 0, 0)
        self.up = vec3(-1, 0, 0)
        # print(self.e.components())
        # print(self.c)
        # print(self.up)

        # Camera coordinate system
        # self.f = (self.c - self.e) / lg.norm((self.c - self.e))
        # self.s = np.cross(self.f, self.up)
        # self.u = -1 * np.cross(self.f, self.s)
        # print("f:", self.f)
        # print("s:", self.s)
        # print("u:", self.u)

        # Field of View
        # self.ratio = float(width) / height
        # self.alpha = np.pi / 8
        # self.h = 2 * np.tan(self.alpha)
        # self.w = self.ratio * self.h
        # print(f"h, w: ${self.h}, ${self.w}")

        self.phi = np.pi / 10
        # rotation matrices
        cos_pos = np.cos(self.phi)
        sin_pos = np.sin(self.phi)
        self.rot_mat_pos = np.array([[ cos_pos, 0, sin_pos], 
                                     [   0, 1,   0], 
                                     [-sin_pos, 0, cos_pos]])
        
        cos_neg = np.cos(-self.phi)
        sin_neg = np.sin(-self.phi)
        self.rot_mat_neg = np.array([[ cos_neg, 0, sin_neg], 
                                     [   0, 1,   0], 
                                     [-sin_neg, 0, cos_neg]])

        self.scene = [
                        Sphere(vec3(0, 0.5, 0), 0.2, vec3(0, 1, 0), mirror = 1),
                      Sphere(vec3(0.4, -0.1, 0), 0.2, vec3(1, 0, 0)),
                      Sphere(vec3(-0.4, -0.1, 0), 0.2, vec3(0, 0, 1), mirror = 0),
                        Plane(vec3(0, -0.5, 0), vec3(0, 1, 0), vec3(1, 1, 1)),
                      Triangle(vec3(0, 0.5, 0), vec3(0.4, -0.1, 0), vec3(-0.4, -0.1, 0), vec3(1, 1, 0))]
        
        r = float(self.width) / self.height
        self.S = (-1, 1 / r + .25, 1, -1 / r + .25)

        # Triangle(vec3(0, 0.5, 0), vec3(0.4, -0.1, 0), vec3(-0.4, -0.1, 0), vec3(1, 1, 0))
        # Triangle(vec3(0, 0.5, -0.5), vec3(0.4, -0.1, -0.5), vec3(-0.4, -0.1, -0.5), vec3(1, 1, 0)),

    def resize(self, new_width, new_height):
        self.width  = new_width
        self.height = new_height
        r = float(self.width) / self.height
        self.S = (-1, 1 / r + .25, 1, -1 / r + .25)

    def rotate_pos(self):

        rt3.rotateLight(self.rot_mat_pos)

        for s in self.scene:
            s.rotate(self.rot_mat_pos)

    def rotate_neg(self):

        rt3.rotateLight(self.rot_mat_neg)
        
        for s in self.scene:
            s.rotate(self.rot_mat_neg)

    def render(self):
        
        x = np.tile(np.linspace(self.S[0], self.S[2], self.width), self.height)
        y = np.repeat(np.linspace(self.S[1], self.S[3], self.height), self.width)

        Q = vec3(x, y, 0)
        color = raytrace(self.e, (Q - self.e).norm(), self.scene, bounce=0)

        rgb = [Image.fromarray((255 * np.clip(c, 0, 1).reshape((self.height, self.width))).astype(np.uint8), "L") for c in color.components()]
        im = Image.merge("RGB", rgb)#.save("rt3.png")
        #im.show()
        return np.array(im)

def raytrace(O, D, scene, bounce = 0):
    # O is the ray origin, D is the normalized ray direction
    # scene is a list of Sphere objects (see below)
    # bounce is the number of the bounce, starting at zero for camera rays

    distances = [s.intersect(O, D) for s in scene]
    nearest = reduce(np.minimum, distances)
    color = rgb(0, 0, 0)
    for (s, d) in zip(scene, distances):
        hit = (nearest != FARAWAY) & (d == nearest)
        if np.any(hit):
            dc = extract(hit, d)
            Oc = O.extract(hit)
            Dc = D.extract(hit)
            cc = s.light(Oc, Dc, dc, scene, bounce)
            color += cc.place(hit)
    return color

def extract(cond, x):
    if isinstance(x, numbers.Number):
        return x
    else:
        return np.extract(cond, x)

# main function
if __name__ == '__main__':

    # set size of render viewport
    width, height = 640, 640

    # instantiate a ray tracer
    ray_tracer = RayTracer(width, height)

    # instantiate a scene
    scene = Scene(width, height, ray_tracer, "Python Raytracer")

    # pass the scene to a render window
    rw = RenderWindow(scene)

    # ... and start main loop
    rw.run()
