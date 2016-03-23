
///
/// 2x2 binning
///

__kernel void binning2(
    __global float * input,
    __global float * output,
    int Nr,
    int Nc)
{
    int gidx = (int) get_global_id(0); // fast dim
    int gidy = (int) get_global_id(1); // slow dim

    if (gidy < Nr && gidx < Nc) {
        int Nc2 = Nc*2;
        float a = input[(gidy*2)*Nc2 + (gidx*2)];
        float b = input[(gidy*2)*Nc2 + (gidx*2+1)];
        float c = input[(gidy*2+1)*Nc2 + (gidx*2)];
        float d = input[(gidy*2+1)*Nc2 + (gidx*2+1)];
        output[gidy*Nc + gidx] = 0.25*(a+b+c+d);
    }
}
