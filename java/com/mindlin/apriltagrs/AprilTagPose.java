package com.mindlin.apriltagrs;

import java.util.Objects;

import com.mindlin.apriltagrs.util.Matrix33;
import com.mindlin.apriltagrs.util.Vec3;

public class AprilTagPose {
    public final Matrix33 R;
    public final Vec3 t;
    public final double error;

    public AprilTagPose(Matrix33 R, Vec3 t, double error) {
        this.R = Objects.requireNonNull(R);
        this.t = Objects.requireNonNull(t);
        this.error = error;
    }
}
