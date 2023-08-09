# this files defines transformations (rotaitons and reflection currently) for different environment 
# it also contains a base class to easily allow extending to other environments
import numpy as np
from parameters import args
from transformation import Transformation, TransformType


def getEnvTransform(env_name): 
    if env_name == "InvertedPendulum-v4": 
        return InvertedPendulumTransforms()
    elif env_name == "Reacher-v4":
        return ReacherTransforms()
    elif env_name == "Pusher-v4":
        return PusherTransforms()
    else: 
        raise NotImplementedError("No transformation for", env_name)

class EnvTransformation: 
    def getComparable(self, a):
        raise NotImplementedError("Comparable not implemented for this environment")
    def canReflect(self, a, b):
        raise NotImplementedError("Reflection not implemented for this environment")
    
    def findRotation(self, a, b):
        raise NotImplementedError("Rotation not implemented for this environment")
    
    def applyTransformation(self, a, t):
        raise NotImplementedError("Transformation not implemented for this environment")

    def rotate(self, a, r):
        raise NotImplementedError("Rotation not implemented for this environment")
    
    def reflect(self, a): 
        raise NotImplementedError("Reflection not implemented for this environment")
    def matrixFromAngle(self, angle):
        return np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
            ])

class InvertedPendulumTransforms(EnvTransformation):
    def getComparable(self, a): 
        return np.array([a[0], np.cos(a[1])]) # cos since its a vertical angle
        # return np.array([a[0], np.rad2deg(a[1])]) 

    def findRotation(self, a, b):
        assert a.shape == b.shape == (4,)
        x1, y1 = self.getComparable(a)
        x2, y2 = self.getComparable(b)

        dot = x1* x2 + y1 * y2
        det = x1 * y2 - y1 * x2
        angle = np.arctan2(det, dot)
        
        if np.isclose(np.linalg.norm(a), np.linalg.norm(b), atol=args.epsilon, rtol=0): 
            return self.matrixFromAngle(angle), angle 
        else: 
            return None, angle
    

    def canReflect(self, a, b):
        assert a.shape == b.shape == (4,)
        a = self.reflect(a)
        return np.allclose(a, b, atol=args.epsilon, rtol=0)
        

    def applyTransform(self, a, t): # rotation currently doesn't do anything to velocities, is that correct?
        assert a.shape == (4,) 

        if t.type == TransformType.ROTATION: 
            x, y = self.getComparable(a)
            point = np.array([x, y])
            point = t.matrix @ point
            return point 
        elif t.type == TransformType.YREFLECTION: 
            return self.getComparable(self.yreflect(a))
        elif t.type == TransformType.XREFLECTION:
            return self.getComparable(self.xreflect(a))
        elif t.type == TransformType.ZREFLECTION:
            return None
        
    def yreflect(self, a): 
        assert a.shape == (4,) 
        return np.array([
                -a[0],
                # (np.pi - a[1]) % (2 * np.pi),
                -a[1],
                -a[2],
                -a[3]
            ]) 

    def xreflect(self, a):
        assert a.shape == (4,) 
        angle = a[1]
        if angle > 0: 
            angle = np.pi + angle
        elif angle < 0:
            angle = angle -np.pi
        
        return np.array([
                a[0],
                angle,
                -a[2],
                -a[3]
        ])

class ReacherTransforms(EnvTransformation):
    def getComparable(self, a):
        return np.array([
            a[4], # target position
            a[5],
            a[8], # distance vector
            a[9] 
        ])
    
    def findRotation(self, a, b):
        assert a.shape == b.shape == (11,)

        # starting with the assumption that only rotations of the distance vector and the goal 
        # are relevant. rotating the cos/sin gets weird because they are local and don't
        # help us know if we can rotate the whole world. 

        # rotate goal
        target1 = np.array([a[4], a[5]])
        target2 = np.array([b[4], b[5]])
        
        dot = np.dot(target1, target2)
        det = target1[0] * target2[1] - target1[1] * target2[0]
        
        angle1 = np.arctan2(det, dot)

        # rotate distance vector
        dist1 = np.array([a[8], a[9]])
        dist2 = np.array([b[8], b[9]])

        dot = np.dot(dist1, dist2)
        det = dist1[0] * dist2[1] - dist1[1] * dist2[0]
        angle2 = np.arctan2(det, dot)


        goalsCLose = np.isclose(np.linalg.norm(target1), np.linalg.norm(target2), atol=args.epsilon, rtol=0)
        distClose = np.isclose(np.linalg.norm(dist1), np.linalg.norm(dist2), atol=args.epsilon, rtol=0)
        anglesClose = np.isclose(angle1, angle2, atol=args.epsilon, rtol=0)
        if goalsCLose and distClose and anglesClose: 
            return self.matrixFromAngle(angle1), angle1
        else: 
            return None, angle1
        
    def applyTransform(self, a, t): # rotation currently doesn't do anything to velocities, is that correct?
        assert a.shape == (11,) 

        if t.type == TransformType.ROTATION: 
            point= self.getComparable(a)
            target = point[:2]
            dist = point[2:]

            target = t.matrix @ target
            dist = t.matrix @ dist
            point = np.concatenate([target, dist])
            return point 
        elif t.type == TransformType.YREFLECTION: 
            return self.getComparable(self.yreflect(a))
        elif t.type == TransformType.XREFLECTION:
            return self.getComparable(self.xreflect(a))
        elif t.type == TransformType.ZREFLECTION:
            return None 
        
    def xreflect(self, a): 
        assert a.shape == (11,) 
        b = a
        b[4] = -b[4]
        b[8] = -b[8]
        return b
        
    def yreflect(self, a): 
        assert a.shape == (11,) 
        b = a
        b[5] = -b[5]
        b[9] = -b[9]
        return b
        
class PusherTransforms(EnvTransformation): 
    def getComparable(self, a): # only looking at positions of fingertip, object, and goal 
        assert a.shape == (23,)
        return np.array([
            a[14], # fingertip position 
            a[15],
            a[16], 
            a[17], # object position
            a[18],
            a[19],
            a[20], # goal position
            a[21], 
            a[22]
        ])

    def findRotation(self, a, b):
        assert a.shape == (23,)
        fingertip1 = a[14:17]
        fingertip2 = b[14:17]
        object1 = a[17:20]
        object2 = b[17:20]
        goal1 = a[20:23]
        goal2 = b[20:23]

        allClose = True 
        allCose = allClose and np.isclose(np.linalg.norm(fingertip1), np.linalg.norm(fingertip2), atol=args.epsilon, rtol=0)
        allClose = allClose and np.isclose(np.linalg.norm(object1), np.linalg.norm(object2), atol=args.epsilon, rtol=0)
        allClose = allClose and np.isclose(np.linalg.norm(goal1), np.linalg.norm(goal2), atol=args.epsilon, rtol=0)
        if not allClose:
            return None, None
        
        frot = self.rotation_matrix_from_vectors(fingertip1, fingertip2)
        orot = self.rotation_matrix_from_vectors(object1, object2)
        grot = self.rotation_matrix_from_vectors(goal1, goal2)

        if np.allclose(frot, orot, atol=args.epsilon, rtol=0) and np.allclose(frot, grot, atol=args.epsilon, rtol=0):
            return frot, 0 # not sure how to get angle at the moment 
        else: 
            return None, None 

    def rotation_matrix_from_vectors(self, vec1, vec2): # from https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        if any(v):
            c = np.dot(a, b)
            s = np.linalg.norm(v)
            kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        else: 
            return np.eye(3)

    def applyTransform(self, a, t): # rotation currently doesn't do anything to velocities, is that correct?
        assert a.shape == (23,)
        if t.type == TransformType.ROTATION: 
            fingertip = a[14:17]
            object = a[17:20]
            goal = a[20:23]

            fingertip = t.matrix @ fingertip
            object = t.matrix @ object
            goal = t.matrix @ goal
            return np.concatenate([fingertip, object, goal])
        elif t.type == TransformType.XREFLECTION: 
            return self.xreflect(a)
        elif t.type == TransformType.YREFLECTION:
            return self.yreflect(a)
        elif t.type == TransformType.ZREFLECTION:
            return self.zreflect(a)


    def xreflect(self, a):
        assert a.shape == (23,)
        b = a 
        b[15] = -b[15] # fingertip y 
        b[16] = -b[16] # fnger tip z
        b[18] = -b[18] # object y
        b[19] = -b[19] # object z
        b[21] = -b[21] # goal y
        b[22] = -b[22] # goal z
        return b 

    def yreflect(self, a):
        assert a.shape == (23,)
        b = a
        b[14] = -b[14] # fingertip x
        b[16] = -b[16] # fingertip z
        b[17] = -b[17] # object x
        b[19] = -b[19] # object z
        b[20] = -b[20] # goal x
        b[22] = -b[22] # goal z
        return b 
    
    def zreflect(self, a): 
        assert a.shape == (23,)
        b = a
        b[14] = -b[14] # fingertip x
        b[15] = -b[15] # fingertip y
        b[17] = -b[17] # object x
        b[18] = -b[18] # object y
        b[20] = -b[20] # goal x
        b[21] = -b[21] # goal y
        return b