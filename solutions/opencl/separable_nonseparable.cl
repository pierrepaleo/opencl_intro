
///
/// Horizontal convolution (along fast dim)
///

__kernel void horizontal_convolution(
    const __global float * input,
    __global float * output,
    __global float * filter,
    int hlen,
    int IMAGE_W,
    int IMAGE_H)
{
    int gidy = (int) get_global_id(1);
    int gidx = (int) get_global_id(0); // fast dim

    if (gidy < IMAGE_H && gidx < IMAGE_W) {

        int c, hL, hR;
        if (hlen & 1) { // odd kernel size
            c = hlen/2;
            hL = c;
            hR = c;
        }
        else { // even kernel size : center is shifted to the left
            c = hlen/2 - 1;
            hL = c;
            hR = c+1;
        }
        int jx1 = c - gidx;
        int jx2 = IMAGE_W - 1 - gidx + c;
        float sum = 0.0f;

        // Convolution with boundaries extension
        for (int jx = 0; jx <= hR+hL; jx++) {
            int idx_x = gidx - c + jx;
            if (jx < jx1) idx_x = jx1-jx-1;
            if (jx > jx2) idx_x = IMAGE_W - (jx-jx2);

            sum += input[(gidy)*IMAGE_W + idx_x] * filter[hlen-1 - jx];
        }
        output[(gidy)*IMAGE_W + gidx] =  sum;
    }
}


///
/// Vertical convolution
///

__kernel void vertical_convolution(
    const __global float * input,
    __global float * output,
    __global float * filter,
    int hlen,
    int IMAGE_W,
    int IMAGE_H)
{
    int gidy = (int) get_global_id(1);
    int gidx = (int) get_global_id(0); // fast dim

    if (gidy < IMAGE_H && gidx < IMAGE_W) {

        int c, hL, hR;
        if (hlen & 1) { // odd kernel size
            c = hlen/2;
            hL = c;
            hR = c;
        }
        else { // even kernel size : center is shifted to the left
            c = hlen/2 - 1;
            hL = c;
            hR = c+1;
        }
        int jy1 = c - gidy;
        int jy2 = IMAGE_H - 1 - gidy + c;
        float sum = 0.0f;

        // Convolution with boundaries extension
        for (int jy = 0; jy <= hR+hL; jy++) {
            int idx_y = gidy - c + jy;
            if (jy < jy1) idx_y = jy1-jy-1;
            if (jy > jy2) idx_y = IMAGE_H - (jy-jy2);

            sum += input[(idx_y)*IMAGE_W + gidx] * filter[hlen-1 - jy];
        }
        output[(gidy)*IMAGE_W + gidx] =  sum;
    }
}



__kernel void nonseparable_convolution(
    const __global float * input,
    __global float * output,
    __global float * filter,
    int hlen,
    int IMAGE_W,
    int IMAGE_H)
{
    int gidy = (int) get_global_id(1);
    int gidx = (int) get_global_id(0); // fast dim

    if (gidy < IMAGE_H && gidx < IMAGE_W) {

        int c, hL, hR;
        if (hlen & 1) { // odd kernel size
            c = hlen/2;
            hL = c;
            hR = c;
        }
        else { // even kernel size : center is shifted to the left
            c = hlen/2 - 1;
            hL = c;
            hR = c+1;
        }


        int jx1 = c - gidx;
        int jx2 = IMAGE_W - 1 - gidx + c;
        int jy1 = c - gidy;
        int jy2 = IMAGE_H - 1 - gidy + c;
        float sum = 0.0f;

        // Convolution with boundaries extension
        for (int jy = 0; jy <= hR+hL; jy++) {
            int idx_y = gidy - c + jy;
            if (jy < jy1) idx_y = jy1-jy-1;
            if (jy > jy2) idx_y = IMAGE_H - (jy-jy2);

            for (int jx = 0; jx <= hR+hL; jx++) {
                int idx_x = gidx - c + jx;
                if (jx < jx1) idx_x = jx1-jx-1;
                if (jx > jx2) idx_x = IMAGE_W - (jx-jx2);

                sum += input[(idx_y)*IMAGE_W + idx_x] * filter[(hlen-1 - jy)*hlen + (hlen-1-jx)];
            }
        }
        output[(gidy)*IMAGE_W + gidx] =  sum;
    }
}






