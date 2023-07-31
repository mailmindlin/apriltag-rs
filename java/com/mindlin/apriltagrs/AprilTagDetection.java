package com.mindlin.apriltagrs;

import com.mindlin.apriltagrs.util.Matrix33;

public abstract class AprilTagDetection {
    AprilTagDetection() {
    }

    public abstract AprilTagFamily getFamily();

    /**
     * Get the decoded tag ID
     */
    public abstract int getTagId();

    /**
     * Get the number of error bits that were corrected
     */
    public abstract int getHammingDistance();

    public abstract float getDecisionMargin();

    public abstract Matrix33 getHomography();
}