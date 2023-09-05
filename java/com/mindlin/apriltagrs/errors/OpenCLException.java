package com.mindlin.apriltagrs.errors;

public class OpenCLException extends AprilTagDetectorBuildException {
    public OpenCLException() {
        super();
    }
    public OpenCLException(String msg) {
        super(msg);
    }
    public OpenCLException(Throwable cause) {
        super(cause);
    }
    public OpenCLException(String msg, Throwable cause) {
        super(msg, cause);
    }
}
