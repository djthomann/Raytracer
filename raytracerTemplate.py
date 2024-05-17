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

from rendering import Scene, RenderWindow
import numpy as np
import numpy.linalg as lg
from rt3 import vec3, Sphere

class RayTracer:

    def __init__(self, width, height):
        self.width  = width
        self.height = height

        # Scene info
        self.e = np.array([0, 0, 1])
        self.c = np.array([0, 0, 0])
        self.up = np.array([1, 0, 0])
        print(self.e)
        # print(self.c)
        # print(self.up)

        # Camera coordinate system
        self.f = (self.c - self.e) / lg.norm((self.c - self.e))
        self.s = np.cross(self.f, self.up)
        self.u = -1 * np.cross(self.f, self.s)
        # print("f:", self.f)
        # print("s:", self.s)
        # print("u:", self.u)

        # Field of View
        self.ratio = width / height
        self.alpha = np.pi / 8
        self.phi = np.pi / 10
        self.h = 2 * np.tan(self.alpha)
        self.w = self.ratio * self.h
        # print(f"h, w: ${self.h}, ${self.w}")

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

        self.scene = [Sphere(vec3(0, 0, 1), 0.2, vec3(0, 0, 1))]

    def resize(self, new_width, new_height):
        self.width  = new_width
        self.height = new_height
        # TODO: modify scene accordingly

    def rotate_pos(self):
        # print("rotate pos")
        self.e = np.dot(self.rot_mat_pos, self.e)
        # print(self.e)
        # TODO: Update camera coordinate system

    def rotate_neg(self):
        # print("rotate neg")
        self.e = np.dot(self.rot_mat_neg, self.e)
        # print(self.e)
        # TODO: Update camera coordinate system

    def render(self):
        # TODO: Replace Dummy Data with Ray Traced Data
        return np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)



# main function
if __name__ == '__main__':

    # set size of render viewport
    width, height = 640, 480

    # instantiate a ray tracer
    ray_tracer = RayTracer(width, height)

    # instantiate a scene
    scene = Scene(width, height, ray_tracer, "Raytracing Template")

    # pass the scene to a render window
    rw = RenderWindow(scene)

    # ... and start main loop
    rw.run()
