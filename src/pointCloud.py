import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import time
import open3d as o3d
from open3d import *
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from IPython import embed
import glm
# import pybullet_data
# import os

def getPointCloud(target, extraCamPos=None):
    """ Creates point cloud of the scene using the syntetic cameras"""
    camTargetPos =  target            #self.simParams["object_position"]  ### position of the object you cared about...
    cameraUp = [0, 0, 1]
    cameraPos = [0.5, 1, 0.7]       #self.simParams["camera_pos"]
    extraCamPos = [1.5, -1, 0.7]    #self.simParams["camera_pos_2"]

    pitch = -20.0   #self.simParams["pitch"]
    yaw = 0         #self.simParams["yaw"]
    roll = 0        #self.simParams["roll"]
    fov = 30        #self.simParams["fov"]

    pixelWidth = 320
    pixelHeight = 200
    aspect = pixelWidth / pixelHeight
    nearPlane = 0.01
    farPlane = 1000

    viewMatrix = p.computeViewMatrix(cameraPos, camTargetPos, cameraUp)
    #viewMatrix = p.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance,yaw, pitch, roll, upAxisIndex)
    projectionMatrix = p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)

    img_arr = p.getCameraImage(pixelWidth, pixelHeight, viewMatrix, projectionMatrix, shadow=1, lightDirection=[
                                1, 1, 1], renderer=p.ER_BULLET_HARDWARE_OPENGL)

    w = img_arr[0]  # width of the image, in pixels
    h = img_arr[1]  # height of the image, in pixels
    rgbBuffer = img_arr[2]  # color data RGB
    depthBuffer = img_arr[3]  # depth data
    maskBuffer = img_arr[4] # mask index data

    maskBuffer  = np.reshape(maskBuffer, (h, w))
    depthBuffer = np.reshape(depthBuffer, (h, w))
    rgb         = np.reshape(rgbBuffer, (h, w, 4))
    rgb = rgb * (1. / 255.)

    try:
        plt.imshow(rgb)
        plt.pause(5)  # show it for 5 secs
        plt.close()
    except: 
        pass


    object_index = []
    #Choose type of mesh reconstruction method
    mesh_type = "Poisson" # "pivoting_ball" or "Poisson"

    points = []
    objPoints = []
    objColor = [] 
    imgH = h
    imgW = w
    stepX = 1
    stepY = 1

    viewport = glm.vec4(0, 0, w, h)

    modelView = glm.mat4(viewMatrix[0], viewMatrix[1], viewMatrix[2], viewMatrix[3],
                            viewMatrix[4], viewMatrix[5], viewMatrix[6], viewMatrix[7],
                            viewMatrix[8], viewMatrix[9], viewMatrix[10], viewMatrix[11],
                            viewMatrix[12], viewMatrix[13], viewMatrix[14], viewMatrix[15],)

    modelProj = glm.mat4(projectionMatrix[0], projectionMatrix[1], projectionMatrix[2], projectionMatrix[3],
                            projectionMatrix[4], projectionMatrix[5], projectionMatrix[6], projectionMatrix[7],
                            projectionMatrix[8], projectionMatrix[9], projectionMatrix[10], projectionMatrix[11],
                            projectionMatrix[12], projectionMatrix[13], projectionMatrix[14], projectionMatrix[15])

    colors = []
    count = 0
    for hh in range(0, imgH, stepX):
        for ww in range(0, imgW, stepY):
            depthImg = float(depthBuffer[hh][ww])
            depth = farPlane * nearPlane / \
                (farPlane - (farPlane - nearPlane) * depthImg)
            win = glm.vec3(ww, h-hh, depthImg)
            if depth < farPlane:
                position = glm.unProject(
                    win, modelView, modelProj, viewport)
                #print('position: '+str(position))
                points.append([position[0], position[1], position[2]])
                temp = rgb[hh][ww]
                colors.append([temp[0], temp[1], temp[2]])
                #print(colors)

                count = count+1
                mask_index = 2
                #Set the index of the desierd object
                if maskBuffer[hh][ww] == mask_index:
                    object_index.append(count)
        

    # Camera number two
    if extraCamPos is not None:
        w = img_arr[0]  # width of the image, in pixels    
        
        viewMatrix = p.computeViewMatrix(extraCamPos, camTargetPos, cameraUp)
        
        projectionMatrix = p.computeProjectionMatrixFOV(
            fov, aspect, nearPlane, farPlane)
        img_arr = p.getCameraImage(pixelWidth, pixelHeight, viewMatrix, projectionMatrix, shadow=1, lightDirection=[
                                1, 1, 1], renderer=p.ER_BULLET_HARDWARE_OPENGL)
        w = img_arr[0]  # width of the image, in pixels
        h = img_arr[1]  # height of the image, in pixels
        rgbBuffer = img_arr[2]  # color data RGB
        depthBuffer = img_arr[3]  # depth data
        maskBuffer = img_arr[4] # mask index data

        maskBuffer  = np.reshape(maskBuffer, (h, w))
        depthBuffer = np.reshape(depthBuffer, (h, w))
        rgb         = np.reshape(rgbBuffer, (h, w, 4))
        rgb = rgb * (1. / 255.)

        modelView = glm.mat4(viewMatrix[0], viewMatrix[1], viewMatrix[2], viewMatrix[3],
                            viewMatrix[4], viewMatrix[5], viewMatrix[6], viewMatrix[7],
                            viewMatrix[8], viewMatrix[9], viewMatrix[10], viewMatrix[11],
                            viewMatrix[12], viewMatrix[13], viewMatrix[14], viewMatrix[15],)
        modelProj = glm.mat4(projectionMatrix[0], projectionMatrix[1], projectionMatrix[2], projectionMatrix[3],
                            projectionMatrix[4], projectionMatrix[5], projectionMatrix[6], projectionMatrix[7],
                            projectionMatrix[8], projectionMatrix[9], projectionMatrix[10], projectionMatrix[11],
                            projectionMatrix[12], projectionMatrix[13], projectionMatrix[14], projectionMatrix[15])

    for hh in range(0, imgH, stepX):
        for ww in range(0, imgW, stepY):
            depthImg = float(depthBuffer[hh][ww])
            depth = farPlane * nearPlane / \
                (farPlane - (farPlane - nearPlane) * depthImg)
            win = glm.vec3(ww, h-hh, depthImg)
            if depth < farPlane:
                position = glm.unProject(
                    win, modelView, modelProj, viewport)
                # print('position: '+str(position))
                points.append([position[0], position[1], position[2]])
                temp = rgb[hh][ww]                   
                colors.append([temp[0], temp[1], temp[2]])
                count = count+1

                #Set the index of the desierd object
                mask_index = 2
                if maskBuffer[hh][ww] == mask_index:
                    object_index.append(count)
    
    # Extracting target object poins and colors 
    for i in range(len(object_index)-1):
        objPoints.append(points[object_index[i]])
        objColor.append(colors[object_index[i]])
    
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.cpu.pybind.utility.Vector3dVector(np.array(points))
    pcl.colors = o3d.cpu.pybind.utility.Vector3dVector(np.array(colors))
    
    #filename = "generated_pointcloud.ply"
    #val = o3d.io.write_point_cloud(filename, pcl)
    #print(f"Generated point cloud : {val}")

    o3d.visualization.draw_geometries([pcl])
    
    print('Length of target object points array')
    print(len(objPoints))
    print(len(objColor))

    # Poisson mesh method
    #-------------------------------------------------------------------------------------
    if len(objPoints) > 0 and len(objColor) > 0 and mesh_type == "Poisson":

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.cpu.pybind.utility.Vector3dVector(np.array(objPoints))
        pcd.colors = o3d.cpu.pybind.utility.Vector3dVector(np.array(objColor))    

        o3d.visualization.draw_geometries([pcd])
        #pcd = o3d.io.write_point_cloud(filename2, pcd)

        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(100)
    
        # Bottle
        poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=12, width=0, scale=1.3, linear_fit=True)[0]
        
        # sphere
        #poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=12, width=0, scale=3.7, linear_fit=True)[0]

        # Skrew driver
        #poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=12, width=0, scale=1.5, linear_fit=False)[0]

        # cube
        #poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=12, width=0, scale=3.3, linear_fit=True)[0]
        
        # teddy bear
        #poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10, width=0, scale=2.2, linear_fit=True)[0]

        #bbox = pcd.get_axis_aligned_bounding_box()
        #p_mesh_crop = poisson_mesh.crop(bbox)

        poisson_mesh.translate((-1,0,-0.06))
        #R = poisson_mesh.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0))
        #poisson_mesh.rotate(R)

        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        """
        mesh_tx = copy.deepcopy(mesh).translate((1, 0, 0))
        mesh_ty = copy.deepcopy(mesh).translate((0, 1.3, 0))
        print(f'Center of mesh tx: {mesh_tx.get_center()}')
        print(f'Center of mesh: {mesh.get_center()}')
        print(f'Center of mesh ty: {mesh_ty.get_center()}')
        """
        
        pcd.paint_uniform_color([1, 0.706, 0])
        o3d.visualization.draw_geometries([pcd, poisson_mesh, mesh])
        o3d.visualization.draw_geometries([poisson_mesh])

        #Save mesh as .obj file
        filename2 = "generated_mesh.obj"
        o3d.io.write_triangle_mesh(filename2, poisson_mesh)

    # Pivoting ball mesh method
    #-------------------------------------------------------------------------------------
    if len(objPoints) > 0 and len(objColor) > 0 and mesh_type == "pivoting_ball":
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.cpu.pybind.utility.Vector3dVector(np.array(objPoints))
        pcd.colors = o3d.cpu.pybind.utility.Vector3dVector(np.array(objColor))   

        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.open3d.cpu.pybind.utility.Vector3dVector(
            np.array(points))
        pcl.colors = o3d.open3d.cpu.pybind.utility.Vector3dVector(
            np.array(colors))
        
        print(pcl.points[1])
        filename = "generated_pointcloud.ply"
        val = o3d.io.write_point_cloud(filename, pcl)
        print(f"Generated point cloud : {val}")
        #o3d.visualization.draw_geometries([pcl])
        
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(100)
        
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 3 * avg_dist
        
        #Create mesh with ball-pivoting algorithm
        bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2]))
        
        #Downsample mesh for efficency, tweak as necissary 
        dec_mesh = bpa_mesh.simplify_quadric_decimation(100000)

        dec_mesh.translate((-1, 0,-0.06))
        
        #Experiment with these if the mesh artifacts in places
        dec_mesh.remove_degenerate_triangles()
        dec_mesh.remove_duplicated_triangles()
        dec_mesh.remove_duplicated_vertices()
        dec_mesh.remove_non_manifold_edges()
        
        o3d.visualization.draw_geometries([pcd, dec_mesh])

        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        """
        mesh_tx = copy.deepcopy(mesh).translate((1, 0, 0))
        mesh_ty = copy.deepcopy(mesh).translate((0, 1.3, 0))
        print(f'Center of mesh tx: {mesh_tx.get_center()}')
        print(f'Center of mesh: {mesh.get_center()}')
        print(f'Center of mesh ty: {mesh_ty.get_center()}')
        """
        pcd.paint_uniform_color([1, 0.706, 0])
        o3d.visualization.draw_geometries([pcd, dec_mesh, mesh])
        o3d.visualization.draw_geometries([dec_mesh])

        #Save mesh as .obj file
        filename2 = "generated_mesh.obj"
        o3d.io.write_triangle_mesh(filename2, dec_mesh)