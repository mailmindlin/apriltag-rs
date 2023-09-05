package com.mindlin.apriltagrs.errors;

public class OpenCLUnavailableException extends AprilTagDetectorBuildException {
    public OpenCLUnavailableException() {
        super();
    }
    public OpenCLUnavailableException(String msg) {
        super(msg);
    }
    public OpenCLUnavailableException(Throwable cause) {
        super(cause);
    }
    public OpenCLUnavailableException(String msg, Throwable cause) {
        super(msg, cause);
    }
}
