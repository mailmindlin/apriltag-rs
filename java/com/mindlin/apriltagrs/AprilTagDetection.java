package com.mindlin.apriltagrs;

import com.mindlin.apriltagrs.util.Matrix33;
import com.mindlin.apriltagrs.util.Point2d;

/**
 * A detection of an AprilTag
 */
public abstract class AprilTagDetection {
    AprilTagDetection() {
    }

    /**
     * Get the decoded tag ID
     */
    public abstract int getTagId();

    /**
     * @return The decoded tag's family name.
     */
    public String getFamilyName() {
        return this.getFamily().getName();
    }

    /**
     * @return The decoded tag's family
     */
    public abstract AprilTagFamily getFamily();

    /**
     * @return The center of the detection in image pixel coordinates.
     */
    public abstract Point2d getCenter();

    /**
     * 
     * @return The corners of the tag in image pixel coordinates. These always wrap counter-clock wise around the tag.
     */
    public abstract Point2d[] getCorners();

    /**
     * @return The number of error bits that were corrected
     */
    public abstract int getHammingDistance();

    /**
     * Gets a measure of the quality of the binary decoding process: the average difference between
     * the intensity of a data bit versus the decision threshold. Higher numbers roughly indicate
     * better decodes. This is a reasonable measure of detection accuracy only for very small tags--
     * not effective for larger tags (where we could have sampled anywhere within a bit cell and
     * still gotten a good detection.)
     * 
     * @return Decision margin
     */
    public abstract float getDecisionMargin();

    /**
     * Gets the 3x3 homography matrix describing the projection from an "ideal" tag (with corners
     * at (-1,1), (1,1), (1,-1), and (-1, -1)) to pixels in the image.
     * 
     * @return Homography matrix
     */
    public abstract Matrix33 getHomography();
}