package com.mindlin.apriltagrs.errors;

public class AccelerationNotAvailableException extends AprilTagDetectorBuildException {
    public AccelerationNotAvailableException() {
        super();
    }
    public AccelerationNotAvailableException(String msg) {
        super(msg);
    }
    public AccelerationNotAvailableException(Throwable cause) {
        super(cause);
    }
    public AccelerationNotAvailableException(String msg, Throwable cause) {
        super(msg, cause);
    }
}
