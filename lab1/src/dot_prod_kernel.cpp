#include <assert.h>
#define DATA_SIZE 4096

extern "C" {

void load(float a_buf[DATA_SIZE], const float *a_in,
          float b_buf[DATA_SIZE], const float *b_in) {
#pragma HLS inline off
    for (int i = 0; i < DATA_SIZE; i++) {
#pragma HLS pipeline
        a_buf[i] = a_in[i];
    }
    for (int i = 0; i < DATA_SIZE; i++) {
        b_buf[i] = b_in[i];
    }
}

void store(float c_buf[1], float *c) {
  *c += c_buf[0];
}

void compute(const float a_buf[DATA_SIZE], const float b_buf[DATA_SIZE], float c_buf[DATA_SIZE]) {
    float a_buf_normal[DATA_SIZE], b_buf_normal[DATA_SIZE];
#pragma HLS array_partition variable=a_buf_normal block factor=8
#pragma HLS array_partition variable=b_buf_normal block factor=8
    float c_buf_normal[1];
    c_buf_normal[0] = 0.f;

    copy_a_buf: for (int i = 0; i < DATA_SIZE; i++) {
        a_buf_normal[i] = a_buf[i];
    }

    copy_b_buf: for (int i = 0; i < DATA_SIZE; i++) {
        b_buf_normal[i] = b_buf[i];
    }

    calc_add: for (int i = 0; i < DATA_SIZE; i++) {
#pragma HLS unroll factor=2
        c_buf_normal[0] += a_buf_normal[i] * b_buf_normal[i];
    }

    copy_c_buf: c_buf[0] = c_buf_normal[0];

    return;
}


void dot_prod_kernel(const float *a, const float *b, float *c, const int num_elems) {
  /********  you can change AXI bus width  **********/
#pragma HLS interface m_axi port = a offset = slave bundle = gmem
#pragma HLS interface m_axi port = b offset = slave bundle = gmem
#pragma HLS interface m_axi port = c offset = slave bundle = gmem
#pragma HLS interface s_axilite port = a bundle = control
#pragma HLS interface s_axilite port = b bundle = control
#pragma HLS interface s_axilite port = c bundle = control
#pragma HLS interface s_axilite port = num_elems bundle = control
#pragma HLS interface s_axilite port = return bundle = control
  assert(num_elems <= 4096); // this helps HLS estimate the loop trip count
  /***************************
   * your code goes here ... *
   ***************************/
  float a_buf[DATA_SIZE], b_buf[DATA_SIZE];
  float c_buf[1];

  int num_full_loops = num_elems / DATA_SIZE;

  for (int i = 0; i < num_full_loops; i++) {
#pragma HLS loop_tripcount min=15 max=16
    load(a_buf, a + (DATA_SIZE * i), b_buf, b + (DATA_SIZE * i));
    compute(a_buf, b_buf, c_buf);
    store(c_buf, c);
  } 
   
}

} // extern "C"
