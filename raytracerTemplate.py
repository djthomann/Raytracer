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
from rt3 import FARAWAY
from functools import reduce
import numbers
from vec3 import vec3

rgb = vec3
L = vec3(0, 5, 5)   
EYE = vec3(0, 0, 5) 

class RayTracer:

    def __init__(self, width, height):
        self.width  = width
        self.height = height

        # Scene info
        self.e = EYE
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
        self.ratio = float(width) / height
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

        self.scene = [Sphere(vec3(0, 0.5, 0), 0.2, vec3(0, 1, 0)),
                      Sphere(vec3(0.4, -0.1, 0), 0.2, vec3(1, 0, 0)),
                      Sphere(vec3(-0.4, -0.1, 0), 0.2, vec3(0, 0, 1)),
                      Plane(vec3(0, -0.5, 0), vec3(0, 1, 0), vec3(1, 1, 1)),
                      Triangle(vec3(0, 0.5, 0), vec3(0.4, -0.1, 0), vec3(-0.4, -0.1, 0), vec3(1, 1, 0))]
        
        # Triangle(vec3(0, 0.5, 0), vec3(0.4, -0.1, 0), vec3(-0.4, -0.1, 0), vec3(1, 1, 0))
        # Triangle(vec3(0, 0.5, -0.5), vec3(0.4, -0.1, -0.5), vec3(-0.4, -0.1, -0.5), vec3(1, 1, 0)),

    def resize(self, new_width, new_height):
        self.width  = new_width
        self.height = new_height
        # TODO: modify scene accordingly

    def rotate_pos(self):
        global L
        L_comp = np.array(L.components())
        L_new_comp = np.dot(self.rot_mat_pos, L_comp)
        L = vec3(L_new_comp[0], L_new_comp[1], L_new_comp[2])

        for s in self.scene:
            s.rotate(self.rot_mat_pos)

    def rotate_neg(self):
        global L
        L_comp = np.array(L.components())
        L_new_comp = np.dot(self.rot_mat_neg, L_comp)
        L = vec3(L_new_comp[0], L_new_comp[1], L_new_comp[2])
        
        for s in self.scene:
            s.rotate(self.rot_mat_neg)

    def render(self):
        
        r = float(self.width) / self.height
        # Screen coordinates: x0, y0, x1, y1.
        # S = (-1, -1 / r + .25, 1, 1 / r + .25)
        S = (-1, 1 / r + .25, 1, -1 / r + .25)
        # p_min = self.f + self.s * (-0.5 * self.w) + self.u * (- 0.5 * self.h)
        # print(p_min)
        x = np.tile(np.linspace(S[0], S[2], self.width), self.height)
        y = np.repeat(np.linspace(S[1], S[3], self.height), self.width)

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
    
class Sphere:
    def __init__(self, center, r, diffuse, mirror = 0.5):
        self.c = center
        self.r = r
        self.diffuse = diffuse
        self.mirror = mirror

    def rotate(self, matrix):
        center_comp = np.array(self.c.components())
        center_new_comp = np.dot(matrix, center_comp)
        self.c = vec3(center_new_comp[0], center_new_comp[1], center_new_comp[2])

    def intersect(self, O, D):
        b = 2 * D.dot(O - self.c)
        c = abs(self.c) + abs(O) - 2 * self.c.dot(O) - (self.r * self.r)
        disc = (b ** 2) - (4 * c)
        sq = np.sqrt(np.maximum(0, disc))
        h0 = (-b - sq) / 2
        h1 = (-b + sq) / 2
        h = np.where((h0 > 0) & (h0 < h1), h0, h1)
        pred = (disc > 0) & (h > 0) # condition for hit
        return np.where(pred, h, FARAWAY)

    def diffusecolor(self, M):
        return self.diffuse

    def light(self, O, D, d, scene, bounce):
        M = (O + D * d)                     # intersection point
        N = (M - self.c) * (1. / self.r)        # normal
        toL = (L - M).norm()                    # direction to light
        toO = (EYE - M).norm()                    # direction to ray origin
        nudged = M + N * .0001                  # M nudged to avoid itself

        # Shadow: find if the point is shadowed or not.
        # This amounts to finding out if M can see the light
        light_distances = [s.intersect(nudged, toL) for s in scene]
        light_nearest = reduce(np.minimum, light_distances)
        seelight = light_distances[scene.index(self)] == light_nearest

        # Ambient
        color = rgb(0.05, 0.05, 0.05)

        # Lambert shading (diffuse)
        lv = np.maximum(N.dot(toL), 0)

        color += self.diffusecolor(M) * lv * seelight

        # Debugging outputs for the first point

        # Reflection
        if bounce < 2:
            rayD = (D - N * 2 * D.dot(N)).norm()
            color += raytrace(nudged, rayD, scene, bounce + 1) * self.mirror

        # Blinn-Phong shading (specular)
        phong = N.dot((toL + toO).norm())
        color += rgb(1, 1, 1) * np.power(np.clip(phong, 0, 1), 50) * seelight
        return color

class Plane:

    def __init__(self, center, normal, diffuse, mirror = 0.5):
        self.center = center
        self.normal = normal
        self.diffuse = diffuse
        self.mirror = mirror

    def rotate(self, matrix):
        pass

    def intersect(self, O, D):
        oben1 = self.center - O
        oben2 = oben1.dot(self.normal)
        unten = D.dot(self.normal)
        t = oben2 / unten
        pred = t > 0
        return np.where(pred, t, FARAWAY)

    def diffusecolor(self, M):
        scale = 2.0
        checker = ((np.floor(M.x * scale).astype(int) % 2) == (np.floor(M.z * scale).astype(int) % 2))
        # checker = ((M.x * 2).astype(int) % 2) == ((M.z * 2).astype(int) % 2)
        return self.diffuse * checker
    
    def light(self, O, D, d, scene, bounce):
        M = (O + D * d)                         # intersection point
        N = self.normal                         # normal
        toL = (L - M).norm()                    # direction to light
        toO = (EYE - M).norm()                    # direction to ray origin
        nudged = M + N * .0001                  # M nudged to avoid itself

        light_distances = [s.intersect(nudged, toL) for s in scene]
        light_nearest = reduce(np.minimum, light_distances)
        seelight = light_distances[scene.index(self)] == light_nearest

        # Ambient
        color = rgb(0.05, 0.05, 0.05)

        # Lambert shading (diffuse)
        lv = np.maximum(N.dot(toL), 0)
        color += self.diffusecolor(M) * lv * seelight

        # Reflection
        if bounce < 2:
            rayD = (D - N * 2 * D.dot(N)).norm()
            color += raytrace(nudged, rayD, scene, bounce + 1) * self.mirror

        # Blinn-Phong shading (specular)
        phong = N.dot((toL + toO).norm())
        color += rgb(1, 1, 1) * np.power(np.clip(phong, 0, 1), 50) * seelight
        return color

class Triangle:

    def __init__(self, a:vec3, b:vec3, c:vec3, diffuse:vec3, mirror = 0.1):
        self.a = a
        self.b = b
        self.c = c
        self.diffuse = diffuse
        print(diffuse.components())
        self.mirror = mirror

        self.u = b - a 
        self.v = c - a
        
        array_u = np.array(self.u.components())
        array_v = np.array(self.v.components())
        normal = np.cross(array_u, array_v)
        normal /= lg.norm(normal)
        self.normal = vec3(normal[0], normal[1], -normal[2])

    def rotate(self, matrix):
        a_comp = np.array(self.a.components())
        a_new_comp = np.dot(matrix, a_comp)
        self.a = vec3(a_new_comp[0], a_new_comp[1], a_new_comp[2])

        b_comp = np.array(self.b.components())
        b_new_comp = np.dot(matrix, b_comp)
        self.b = vec3(b_new_comp[0], b_new_comp[1], b_new_comp[2])

        c_comp = np.array(self.c.components())
        c_new_comp = np.dot(matrix, c_comp)
        self.c = vec3(c_new_comp[0], c_new_comp[1], c_new_comp[2])

        self.u = self.b - self.a 
        self.v = self.c - self.a

        array_u = np.array(self.u.components())
        array_v = np.array(self.v.components())
        normal = np.cross(array_u, array_v)
        normal /= lg.norm(normal)
        self.normal = vec3(normal[0], normal[1], -normal[2])
        print(self.normal.components())

    def intersect(self, O: vec3, D: vec3):
        
        w = O - self.a

        C1 = w.cross(self.u)
        C2 = D.cross(self.v)

        S1 = C2.dot(self.u)
        S2 = C1.dot(self.v)
        S3 = C2.dot(w)
        S4 = C1.dot(D)

        t = S2 / S1
        r = S3 / S1
        s = S4 / S1

        pred =  np.logical_and(np.logical_and(r < 1, r > 0),
                np.logical_and(np.logical_and(s < 1, s > 0), (r + s) <= 1))
        
        return np.where(pred, t, FARAWAY)

    def diffusecolor(self, M):
        return self.diffuse
    
    def light(self, O, D, d, scene, bounce):
        M = (O + D * d)                         # intersection point
        N = self.normal                         # normal
        toL = (L - M).norm()                    # direction to light
        toO = (EYE - M).norm()                  # direction to ray origin
        nudged = M + N * .0001                  # M nudged to avoid itself

        # Shadow: find if the point is shadowed or not.
        # This amounts to finding out if M can see the light
        light_distances = [s.intersect(nudged, toL) for s in scene]
        light_nearest = reduce(np.minimum, light_distances)
        seelight = light_distances[scene.index(self)] == light_nearest
        # print(seelight)

        # Ambient
        color = rgb(0.05, 0.05, 0.05)

        # Lambert shading (diffuse)
        lv = np.maximum(N.dot(toL), 0)
        color += self.diffusecolor(M) * lv * seelight

        # Reflection
        if bounce < 2:
            rayD = (D - N * 2 * D.dot(N)).norm()
            color += raytrace(nudged, rayD, scene, bounce + 1) * self.mirror

        # Blinn-Phong shading (specular)
        phong = N.dot((toL + toO).norm())
        color += rgb(1, 1, 1) * np.power(np.clip(phong, 0, 1), 50) * seelight
        # print(color.components())
        return color

# main function
if __name__ == '__main__':

    # set size of render viewport
    width, height = 640, 640

    # instantiate a ray tracer
    ray_tracer = RayTracer(width, height)

    # instantiate a scene
    scene = Scene(width, height, ray_tracer, "Raytracing Template")

    # pass the scene to a render window
    rw = RenderWindow(scene)

    # ... and start main loop
    rw.run()
