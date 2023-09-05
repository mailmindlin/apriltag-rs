package com.mindlin.apriltagrs;

import java.util.Objects;

import com.mindlin.apriltagrs.util.Matrix33;
import com.mindlin.apriltagrs.util.Vec3;

public final class AprilTagPose {
    public final Matrix33 R;
    public final Vec3 t;
    public final double error;

    public AprilTagPose(Matrix33 R, Vec3 t, double error) {
        this.R = Objects.requireNonNull(R);
        this.t = Objects.requireNonNull(t);
        this.error = error;
    }

    // Used by JNI, the signature makes it easier to deal with
    private AprilTagPose(double R_11, double R_12, double R_13, double R_21, double R_22, double R_23, double R_31, double R_32, double R_33, double t_0, double t_1, double t_2, double error) {
        this(new Matrix33(R_11, R_12, R_13, R_21, R_22, R_23, R_31, R_32, R_33), new Vec3(t_0, t_1, t_2), error);
    }
}
