package com.mindlin.apriltagrs.errors;

public class ThreadPoolBuildException extends AprilTagDetectorBuildException {
    public ThreadPoolBuildException() {
        super();
    }
    public ThreadPoolBuildException(String msg) {
        super(msg);
    }
    public ThreadPoolBuildException(Throwable cause) {
        super(cause);
    }
    public ThreadPoolBuildException(String msg, Throwable cause) {
        super(msg, cause);
    }
}
