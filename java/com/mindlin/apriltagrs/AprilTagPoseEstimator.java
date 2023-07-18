package com.mindlin.apriltagrs;

import java.util.Arrays;
import java.util.Collection;
import java.util.List;

public class AprilTagPoseEstimator {
    public final double tagSize;
    public final double fx;
    public final double fy;
    public final double cx;
    public final double cy;

    private static native AprilTagPose[] nativeEstimatePose(
        double tagSize,
        double fx,
        double fy,
        double cx,
        double cy,
        long detectionPtr
    );

    public AprilTagPoseEstimator(double tagSize, double fx, double fy, double cx, double cy) {
        this.tagSize = tagSize;
        this.fx = fx;
        this.fy = fy;
        this.cx = cx;
        this.cy = cy;
    }

    public List<AprilTagPose> estimatePose(AprilTagDetection detection) {
        var poses = detection.nativeRead(ptr ->
            AprilTagPoseEstimator.nativeEstimatePose(
                this.tagSize,
                this.fx,
                this.fy,
                this.cx,
                this.cy,
                ptr
            )
        );
        return Arrays.asList(poses);
    }

    public List<List<AprilTagPose>> estimatePoses(Collection<AprilTagDetection> detections) {
        return detections
            .stream()
            .map(this::estimatePose)
            .toList();
    }
}
