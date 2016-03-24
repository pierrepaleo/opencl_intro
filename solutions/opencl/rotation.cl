__kernel void rotation(
    const __global float * img,
    __global float * output,
    int IMG_W,
    int IMG_H,
    int centerx,
    int centery,
    float theta)
{
    int gidy = (int) get_global_id(1);
    int gidx = (int) get_global_id(0);

    if (gidy < IMG_H && gidx < IMG_W) {

        float ct = cos(theta);
        float st = sin(theta);

        float x = (gidx-centerx)*ct - (gidy-centery)*st;
        float y = (gidx-centery)*st + (gidy-centerx)*ct;
        x += centerx;
        y += centery;

        if (x < 0 || x > IMG_W-1 || y < 0 || y > IMG_H-1) {
            output[gidy*IMG_W + gidx] = 0;
            return;
        }

        // Bilinear interpolation
        /*
                (xm, ym)        (xp, ym)
                         (x, y)
                (xm, yp)        (xp, yp)
        */
        int xm = (int) floor(x);
        int xp = (int) ceil(x);
        int ym = (int) floor(y);
        int yp = (int) ceil(y);
        float val;
        float tol = 0.001f; // CHECKME
        if ((x - xm) < tol && (y - ym) < tol) val = img[ym*IMG_W + xm];
        else if ((xp - x) < tol && (yp - y) < tol) val = img[yp*IMG_W + xp];
        else {
            // Mirror
            if (xm < 0) xm = 0;
            if (xp >= IMG_W) xp = IMG_W - 1;
            if (ym < 0) ym = 0;
            if (yp >= IMG_H) yp = IMG_H -1;
            // Interp
            val = img[yp*IMG_W+xm]*(xp-x)*(y-ym)
                        + img[yp*IMG_W+xp]*(x-xm)*(y-ym)
                        + img[ym*IMG_W + xm]*(xp-x)*(yp-y)
                        + img[ym*IMG_W + xp]*(x-xm)*(yp-y);

        }
        output[gidy*IMG_W + gidx] = val;
    }
}

