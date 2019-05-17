import numpy as np

def uniform_from_sphere(npts):
    
    phi = 2.0 * np.pi * np.random.uniform(0.0, 1.0, npts)
    cos_theta = 2.0 * np.random.uniform(0.0, 1.0, npts) - 1.0

    V = np.column_stack([cos_theta, phi])

    return V 

def transform_to_cartesian(V):
    
    cos_theta = V[:,0]
    theta = np.arccos(cos_theta)
    phi = V[:,1]

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    V = np.column_stack([x, y, z])    

    return V

def transform_to_spherical(V):
    x = V[:,0]
    y = V[:,1]
    z = V[:,2]

    cos_theta = z
    #phi = np.arctan2(y,x) + np.pi
    #idx = y<0
    #y[idx] = -y[idx]
    phi = np.arctan2(y,x)
    idx = phi<0.0
    phi[idx] = phi[idx] + np.pi * 2.0
    return np.column_stack([cos_theta, phi])
    

