package com.mindlin.apriltagrs.errors;

public class AprilTagDetectorBuildException extends RuntimeException {
    public AprilTagDetectorBuildException() {
        super();
    }
    public AprilTagDetectorBuildException(String msg) {
        super(msg);
    }
    public AprilTagDetectorBuildException(Throwable cause) {
        super(cause);
    }
    public AprilTagDetectorBuildException(String msg, Throwable cause) {
        super(msg, cause);
    }
}
