// Physics simulation library for TinyGrad custom ops
#include <stdint.h>
#include <string.h>
#include <math.h>

// Basic physics data structures
typedef struct {
    float x, y, z;
} Vec3;

typedef struct {
    Vec3 position;
    Vec3 velocity;
    float mass;
    float radius;
} RigidBody;

// Helper functions
static inline Vec3 vec3_add(Vec3 a, Vec3 b) {
    return (Vec3){a.x + b.x, a.y + b.y, a.z + b.z};
}

static inline Vec3 vec3_sub(Vec3 a, Vec3 b) {
    return (Vec3){a.x - b.x, a.y - b.y, a.z - b.z};
}

static inline Vec3 vec3_scale(Vec3 v, float s) {
    return (Vec3){v.x * s, v.y * s, v.z * s};
}

static inline float vec3_dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline float vec3_length_squared(Vec3 v) {
    return vec3_dot(v, v);
}

// Physics integration step - simple Euler integration
// Input format: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, mass, radius]
// Output format: same as input (updated values)
void physics_integrate(float* bodies, int32_t num_bodies, float dt, float* output) {
    // Copy input to output first
    memcpy(output, bodies, num_bodies * 8 * sizeof(float));
    
    // Apply gravity and integrate positions
    const float gravity = -9.81f;
    
    for (int i = 0; i < num_bodies; i++) {
        int idx = i * 8;
        
        // Apply gravity to velocity (only Y component)
        output[idx + 4] += gravity * dt;
        
        // Update positions based on velocities
        output[idx + 0] += output[idx + 3] * dt;  // pos.x += vel.x * dt
        output[idx + 1] += output[idx + 4] * dt;  // pos.y += vel.y * dt
        output[idx + 2] += output[idx + 5] * dt;  // pos.z += vel.z * dt
    }
}

// Collision detection and response
// Input: bodies array, output: contact forces array
void physics_collisions(float* bodies, int32_t num_bodies, float* forces) {
    // Initialize forces to zero
    memset(forces, 0, num_bodies * 3 * sizeof(float));
    
    const float k_spring = 10000.0f;  // Spring constant for penalty method
    const float damping = 0.1f;
    
    for (int i = 0; i < num_bodies; i++) {
        for (int j = i + 1; j < num_bodies; j++) {
            int idx_i = i * 8;
            int idx_j = j * 8;
            
            Vec3 pos_i = {bodies[idx_i], bodies[idx_i + 1], bodies[idx_i + 2]};
            Vec3 pos_j = {bodies[idx_j], bodies[idx_j + 1], bodies[idx_j + 2]};
            Vec3 vel_i = {bodies[idx_i + 3], bodies[idx_i + 4], bodies[idx_i + 5]};
            Vec3 vel_j = {bodies[idx_j + 3], bodies[idx_j + 4], bodies[idx_j + 5]};
            
            float radius_i = bodies[idx_i + 7];
            float radius_j = bodies[idx_j + 7];
            
            Vec3 delta = vec3_sub(pos_j, pos_i);
            float dist_sq = vec3_length_squared(delta);
            float min_dist = radius_i + radius_j;
            
            if (dist_sq < min_dist * min_dist && dist_sq > 0.0001f) {
                float dist = sqrtf(dist_sq);
                float penetration = min_dist - dist;
                
                // Normal vector
                Vec3 normal = vec3_scale(delta, 1.0f / dist);
                
                // Relative velocity
                Vec3 rel_vel = vec3_sub(vel_j, vel_i);
                float vel_along_normal = vec3_dot(rel_vel, normal);
                
                // Apply penalty force
                float force_magnitude = k_spring * penetration - damping * vel_along_normal;
                Vec3 force = vec3_scale(normal, force_magnitude);
                
                // Apply equal and opposite forces
                int force_idx_i = i * 3;
                int force_idx_j = j * 3;
                
                forces[force_idx_i] -= force.x;
                forces[force_idx_i + 1] -= force.y;
                forces[force_idx_i + 2] -= force.z;
                
                forces[force_idx_j] += force.x;
                forces[force_idx_j + 1] += force.y;
                forces[force_idx_j + 2] += force.z;
            }
        }
    }
}

// Combined physics step
// This is the main entry point for TinyGrad custom op
void physics_step(float* bodies, int32_t num_bodies, float dt, float* output) {
    // Allocate temporary force buffer
    float forces[num_bodies * 3];
    
    // Compute collision forces
    physics_collisions(bodies, num_bodies, forces);
    
    // Copy input to output
    memcpy(output, bodies, num_bodies * 8 * sizeof(float));
    
    // Apply forces and integrate
    const float gravity = -9.81f;
    
    for (int i = 0; i < num_bodies; i++) {
        int body_idx = i * 8;
        int force_idx = i * 3;
        
        float mass = output[body_idx + 6];
        if (mass > 0.0f) {
            // Apply forces to velocity
            output[body_idx + 3] += forces[force_idx] / mass * dt;
            output[body_idx + 4] += (forces[force_idx + 1] / mass + gravity) * dt;
            output[body_idx + 5] += forces[force_idx + 2] / mass * dt;
            
            // Update positions
            output[body_idx] += output[body_idx + 3] * dt;
            output[body_idx + 1] += output[body_idx + 4] * dt;
            output[body_idx + 2] += output[body_idx + 5] * dt;
        }
    }
}