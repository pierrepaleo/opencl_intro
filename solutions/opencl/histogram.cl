
__kernel void histogram256(
    __global int * img,
    __global int * hist,
    int Nr,
    int Nc)
{
    int gidx = (int) get_global_id(0); // fast dim
    int gidy = (int) get_global_id(1); // slow dim

    if (gidy < Nr && gidx < Nc) {
        int val = img[gidy*Nc + gidx];
        if (0 <= val && val <= 255) {
            atomic_inc(&(hist[val]));
        }
    }
}
