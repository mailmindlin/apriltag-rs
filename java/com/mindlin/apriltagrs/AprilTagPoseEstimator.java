package com.mindlin.apriltagrs;

public class AprilTagPoseEstimator {
    public final double tagSize;
    public final double fx;
    public final double fy;
    public final double cx;
    public final double cy;

    public AprilTagPoseEstimator(double tagSize, double fx, double fy, double cx, double cy) {
        this.tagSize = tagSize;
        this.fx = fx;
        this.fy = fy;
        this.cx = cx;
        this.cy = cy;
    }

    public AprilTagDetection estimatePose(AprilTagDetection detection) {

    }
}
