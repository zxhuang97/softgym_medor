#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <list>
#include <iterator>
#include <fstream>
#include <string>
#include <map>

#define DIST(p, q) (sqrt((p.x - q.x) * (p.x - q.x) + (p.y - q.y) * (p.y - q.y) + (p.z - q.z) * (p.z - q.z)))

class SoftgymAnyCloth: public Scene
{
public:
    float cam_x;
    float cam_y;
    float cam_z;
    float cam_angle_x;
    float cam_angle_y;
    float cam_angle_z;
    int cam_width;
    int cam_height;
    char cloth_path[200];

    SoftgymAnyCloth(const char* name) : Scene(name) {}

//    char* make_path(char* full_path, std::string path) {
//        strcpy(full_path, getenv("PYFLEXROOT"));
//        strcat(full_path, path.c_str());
//        cout << "mesh path: " << full_path << endl;
//        return full_path;
//    }
//
//    float get_param_float(py::array_t<float> scene_params, int idx)
//    {
//        auto ptr = (float *) scene_params.request().ptr;
//        float out = ptr[idx];
//        return out;
//    }


//    void sortInd(uint32_t* a, uint32_t* b, uint32_t* c)
//    {
//        if (*b < *a)
//            swap(a,b);
//
//        if (*c < *b)
//        {
//            swap(b,c);
//            if (*b < *a)
//                swap(b, a);
//        }
//    }

//
//    void findUnique(map<uint32_t, uint32_t> &unique, Mesh* m)
//    {
//        map<vector<float>, uint32_t> vertex;
//        map<vector<float>, uint32_t>::iterator it;
//
//        uint32_t count = 0;
//        for (uint32_t i=0; i < m->GetNumVertices(); ++i)
//        {
//            Point3& v = m->m_positions[i];
//            float arr[] = {v.x, v.y, v.z};
//            vector<float> p(arr, arr + sizeof(arr)/sizeof(arr[0]));
//
//            it = vertex.find(p);
//            if (it == vertex.end()) {
//                vertex[p] = i;
//                unique[i] = i;
//                count++;
//            }
//            else
//            {
//                unique[i] = it->second;
//            }
//        }
//
//        cout << "total vert:  " << m->GetNumVertices() << endl;
//        cout << "unique vert: " << count << endl;
//    }


    //params ordering: xpos, ypos, zpos, xsize, zsize, stretch, bend, shear
    // render_type, cam_X, cam_y, cam_z, angle_x, angle_y, angle_z, width, height
    void Initialize(py::array_t<float> scene_params = py::array_t<float>(),
     int thread_idx = 0)
    {
        auto ptr = (float *) scene_params.request().ptr;
        float damping = ptr[0];
        float dynamic_friction = ptr[1];
        float particle_friction = ptr[2];
        float gravity = ptr[3];
        float fov = ptr[4];
        float vel = ptr[5];
        float stretchStiffness = ptr[6];
        float shearStiffness = ptr[7];
        float bendStiffness = ptr[8];
        float mass = ptr[9];
        float radius = ptr[10];

        cam_x = ptr[11];
        cam_y = ptr[12];
        cam_z = ptr[13];
        cam_angle_x = ptr[14];
        cam_angle_y = ptr[15];
        cam_angle_z = ptr[16];
        cam_width = int(ptr[17]);
        cam_height = int(ptr[18]);

        int render_type = int(ptr[19]);
        int dimx= int(ptr[20]);
        int dimy= int(ptr[21]);

        int node_num = int(ptr[22]);
        int face_num = int(ptr[23]);

        int stretch_edge_num = int(ptr[24]);
        int bend_edge_num = int(ptr[25]);
        int shear_edge_num = int(ptr[26]);

        int phase = NvFlexMakePhase(0, eNvFlexPhaseSelfCollide | eNvFlexPhaseSelfCollideFilter);

        if (node_num>0){
            float x, y, z;
            int base = 27;
            for (uint32_t i = 0; i < node_num; i++){
                x = float(ptr[base + i * 3]);
                y = float(ptr[base + i * 3 + 1]);
                z = float(ptr[base + i * 3 + 2]);
                // cout << "add point x: " << x << " y: " << y << " z: " << z << endl;
                g_buffers->positions.push_back(Vec4(x, y, z, 1. / mass));
                g_buffers->velocities.push_back(Vec3(vel, vel, vel));
                g_buffers->phases.push_back(phase);
            }
            base = base + node_num * 3;
            for (uint32_t i = 0; i < face_num; i++){
                g_buffers->triangles.push_back(ptr[base + i*3]);
                g_buffers->triangles.push_back(ptr[base + i*3+1]);
                g_buffers->triangles.push_back(ptr[base + i*3+2]);
                auto p1 = g_buffers->positions[ptr[base + i*3]];
                auto p2 = g_buffers->positions[ptr[base + i*3+1]];
                auto p3 = g_buffers->positions[ptr[base + i*3+2]];
                auto U = p2 - p1;
                auto V = p3 - p1;
                auto normal = Vec3(
                    U.y * V.z - U.z * V.y,
                    U.z * V.x - U.x * V.z,
                    U.x * V.y - U.y * V.x);
                g_buffers->triangleNormals.push_back(normal / Length(normal));
            }

            int sender, receiver;
            base = base + face_num*3;
            for (uint32_t i = 0; i < stretch_edge_num; i++ ){
                sender = int(ptr[base + i * 2]);
                receiver = int(ptr[base + i * 2 + 1]);
                CreateSpring(sender, receiver, stretchStiffness); // assume no additional particles are added
            }
            base = base + stretch_edge_num*2;
            for (uint32_t i=0; i< shear_edge_num; i++){
                sender = int(ptr[base + i * 2]);
                receiver = int(ptr[base + i * 2 + 1]);
                CreateSpring(sender, receiver, shearStiffness);
            }

            base += shear_edge_num*2;
            for (uint32_t i = 0; i < bend_edge_num; i++){
                sender = int(ptr[base + i * 2]);
                receiver = int(ptr[base + i * 2 + 1]);
                CreateSpring(sender, receiver, bendStiffness);
            }
        }
        else {
            CreateSpringGrid(Vec3(0, -1, 0), dimx, dimy, 1, radius, phase, stretchStiffness, bendStiffness, shearStiffness, 0.0f, 1.0f / mass);
        }


        g_numSubsteps = 4;
        g_params.numIterations = 30;

        g_params.dynamicFriction = dynamic_friction;
        g_params.particleFriction = particle_friction;
        g_params.damping = damping;
        g_params.sleepThreshold = 0.02f;
        g_params.relaxationFactor = 1.0f;
        g_params.shapeCollisionMargin = 0.04f;
        g_sceneLower = Vec3(-1.0f);
        g_sceneUpper = Vec3(1.0f);

        g_params.radius = 1.8f*radius;
        g_params.collisionDistance = 0.005f;

        g_drawPoints = render_type & 1;
        g_drawCloth = (render_type & 2) >>1;
        g_drawMesh = false;
        g_drawSprings = false;
        g_drawDiffuse = false;

//        bool hasFluids = false;
//        DepthRenderProfile p = {
//            0.f, // minRange
//            5.f // maxRange
//        };
//        if (g_render) // ptr[19] is whether to use a depth sensor
//        {
////            printf("adding a sensor!\n");
//            AddSensor(cam_width, cam_height,  0,  Transform(Vec3(cam_x, cam_y, cam_z),
//            rpy2quat(cam_angle_x, cam_angle_y, cam_angle_z)),  DegToRad(fov), hasFluids, p);
//        }

        // DEBUG
//        cout << "tris: " << g_buffers->triangles.size() << endl;
        //cout << "skinMesh: " << g_meshSkinIndices.size() << endl;
        g_params.gravity[1] = gravity;
    }

    virtual void CenterCamera(void)
    {
        g_camPos = Vec3(cam_x, cam_y, cam_z);
        g_camAngle = Vec3(cam_angle_x, cam_angle_y, cam_angle_z);
        g_screenHeight = cam_height;
        g_screenWidth = cam_width;
    }
};