from PIL import Image
from functools import reduce
import numpy as np
import time
import numbers

def extract(cond, x):
    if isinstance(x, numbers.Number):
        return x
    else:
        return np.extract(cond, x)

class vec3():
    def __init__(self, x, y, z):
        (self.x, self.y, self.z) = (x, y, z)
    def __mul__(self, other):
        return vec3(self.x * other, self.y * other, self.z * other)
    def __add__(self, other):
        return vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    def __sub__(self, other):
        return vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    def dot(self, other):
        return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)
    def cross(self, other):
        return vec3(self.y * other.z - self.z * other.y, self.z*other.x - self.x * other.z, self.x * other.y - self.y * other.x)
    def __abs__(self):
        return self.dot(self)
    def norm(self):
        mag = np.sqrt(abs(self))
        return self * (1.0 / np.where(mag == 0, 1, mag))
    def components(self):
        return (self.x, self.y, self.z)
    def extract(self, cond):
        return vec3(extract(cond, self.x),
                    extract(cond, self.y),
                    extract(cond, self.z))
    def place(self, cond):
        r = vec3(np.zeros(cond.shape), np.zeros(cond.shape), np.zeros(cond.shape))
        np.place(r.x, cond, self.x)
        np.place(r.y, cond, self.y)
        np.place(r.z, cond, self.z)
        return r
rgb = vec3

(w, h) = (640, 400)         # Screen size
L = vec3(0, 5, 5)        # Point light position
E = vec3(0, 0, 5)    # Eye position
FARAWAY = 1.0e39            # an implausibly huge distance

def rotateLight(matrix):
    global L
    L_comp = np.array(L.components())
    L_new_comp = np.dot(matrix, L_comp)
    L = vec3(L_new_comp[0], L_new_comp[1], L_new_comp[2])

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
        toO = (E - M).norm()                    # direction to ray origin
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

class CheckeredSphere(Sphere):
    def diffusecolor(self, M):
        checker = ((M.x * 2).astype(int) % 2) == ((M.z * 2).astype(int) % 2)
        return self.diffuse * checker

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
        # return self.diffuse
    
    def light(self, O, D, d, scene, bounce):
        M = (O + D * d)                         # intersection point
        N = self.normal                         # normal
        toL = (L - M).norm()                    # direction to light
        toO = (E - M).norm()                    # direction to ray origin
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
        self.mirror = mirror

        self.u = b - a 
        self.v = c - a
        
        self.u = self.b - self.a 
        self.v = self.c - self.a

        self.normal = (self.u.cross(self.v)).norm() * -1

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

        self.normal = (self.u.cross(self.v)).norm() * -1

        # print(self.normal.components())

    def diffusecolor(self, M):
        return self.diffuse

    def intersect(self, O, D):


        # Möller–Trumbore Schnittalgorithmus als Alternative

        numerical_error = 0.0001

        # Die Vektoren u und v definieren eine Ebene

        # Test ob der Richtungsvektor des Strahs parallel zur Ebene verläuft
        c1 = D.cross(self.v)
        d1 = self.u.dot(c1)
        mask = (np.abs(d1) > numerical_error)

        if any(mask):
            # Strahl läuft nicht parallel zur Ebene, irgendwo wird geschnitten. Im Dreieck?
            # Strahl läuft parallel --> ff = 0, uu = 0, mask_u = 0, m = 0, mask_v = 0 --> return FARAWAY
            f = np.where(mask, 1.0 / d1, 0.0)

            # Berechne baryzentrischen Koordinatenparameter 1
            s = O - self.a
            u = s.dot(c1) * f
            mask_u = (u >= 0.0) & (u <= 1.0)

            # Berechne baryzentrischen Koordinatenparameter 2
            q = s.cross(self.u)
            m = D.dot(q) * f
            mask_v = (m >= 0.0) & (u + m <= 1.0)

            # Berechne t
            t = self.v.dot(q) * f

            pred =  np.logical_and((np.logical_and(mask, mask_v)), np.logical_and(mask_u, (t > numerical_error)))

            return np.where(pred, t, FARAWAY)
        else:
            # Kein Strahl läuft parallel --> Dreieck wird nicht geschnitten
            return FARAWAY


        # Methode aus der Vorlesung, hat leider nicht funktioniert :/
        # w = O - self.a

        # C1 = w.cross(self.u)
        # C2 = D.cross(self.v)

        # S1 = C2.dot(self.u)
        # S2 = C1.dot(self.v)
        # S3 = C2.dot(w)
        # S4 = C1.dot(D)

        # t = S2 / S1
        # r = S3 / S1
        # s = S4 / S1

        # pred =  np.logical_and(np.logical_and(r < 1, r > 0),
               # np.logical_and(np.logical_and(s < 1, s > 0), (r + s) <= 1))
        
        # return np.where(pred, t, FARAWAY)

    def light(self, O, D, d, scene, bounce):
        M = (O + D * d)                         # intersection point
        N = self.normal                         # normal
        toL = (L - M).norm()                    # direction to light
        toO = (E - M).norm()                  # direction to ray origin
        nudged = M + N * .0001                  # M nudged to avoid itself

        # Shadow: find if the point is shadowed or not.
        # This amounts to finding out if M can see the light
        light_distances = [s.intersect(nudged, toL) for s in scene]
        light_nearest = reduce(np.minimum, light_distances)
        seelight = light_distances[scene.index(self)] == light_nearest
        # print(seelight)

        # backside of triangle is not colored / in shadow
        if self.normal.z < 0:
            seelight = 0

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

def test_scene():
    scene = [
        Sphere(vec3(.75, .1, 1), .6, vec3(0, 0, 1)),
        Sphere(vec3(-.75, .1, 2.25), .6, vec3(.5, .223, .5)),
        Sphere(vec3(-2.75, .1, 3.5), .6, vec3(1, .572, .184)),
        CheckeredSphere(vec3(0,-9.5, 0), 9, vec3(.75, .75, .75), 0.25),
        ]

    r = float(w) / h
    # Screen coordinates: x0, y0, x1, y1.
    S = (-1, 1 / r + .25, 1, -1 / r + .25)
    x = np.tile(np.linspace(S[0], S[2], w), h)
    y = np.repeat(np.linspace(S[1], S[3], h), w)

    t0 = time.time()
    Q = vec3(x, y, 0)
    color = raytrace(E, (Q - E).norm(), scene)
    print ("Took", time.time() - t0)

    rgb = [Image.fromarray((255 * np.clip(c, 0, 1).reshape((h, w))).astype(np.uint8), "L") for c in color.components()]
    im = Image.merge("RGB", rgb)#.save("rt3.png")
    #im.show()
    return np.array(im)
