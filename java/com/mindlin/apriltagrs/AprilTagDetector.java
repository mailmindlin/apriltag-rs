package com.mindlin.apriltagrs;
import java.io.Closeable;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;

import com.mindlin.apriltagrs.AprilTagLibrary.NativeObject;

public final class AprilTagDetector extends NativeObject {
    /**
     * Create new detector object
     * 
     * @return Handle to detector
     */
    private static native long create();
    private static native void setParams(long ptr, int nthreads, float quadDecimate, float quadSigma, boolean refineEdges, float decodeSharpening, boolean debug);
    private static native void addFamily(long ptr, long familyPtr, short maxHamming);
    private static native void removeFamily(long ptr, String familyName);
    private static native long clearFamilies(long ptr);
    private static native AprilTagDetections detect(long ptr, ByteBuffer buf, int width, int height, int stride);

    public static class Builder {
        private final Config config;
        private final List<AprilTagFamily> families;

        public Builder() {
            this(new Config(), Collections.emptyList());
        }

        private Builder(Builder src) {
            Objects.requireNonNull(src);
            this.config = new Config(src.config);
            this.families = new ArrayList<>(src.families);
        }

        public Config getConfig() {
            return this.config;
        }

        /**
         * Set number of threads to use
         * @param nthreads Number of threads. 0 means 
         * @return modified builder
         */
        public Builder threads(int nthreads) {
            var res = new Builder(this);
            res.config.nthreads = nthreads;
            return res;
        }

        public Builder addFamily(String familyName) throws IllegalArgumentException {
            return this.addFamily(familyName, 2);
        }

        public Builder addFamily(String familyName, int maxHamming) throws IllegalArgumentException {
            var family = AprilTagFamily.forName(familyName);
            return this.addFamily(family, maxHamming);
        }

        public Builder addFamily(AprilTagFamily family) {
            return this.addFamily(family, 2);
        }

        public Builder addFamily(AprilTagFamily family, int maxHamming) {

        }

        public AprilTagDetector build() {

        }
    }

    public static final class Config {
        int nthreads = 1;
        float quadDecimate = 2.0f;
        float quadSigma = 0.0f;
        boolean refineEdges = true;
        float decodeSharpening = 0.25f;
        Path debug = null;

        public Config() {
        }

        public Config(Config config) {
            this.nthreads = config.nthreads;
            this.quadDecimate = config.quadDecimate;
            this.quadSigma = config.quadSigma;
            this.refineEdges = config.refineEdges;
            this.decodeSharpening = config.decodeSharpening;
            this.debug = config.debug;
        }

        public int getNumThreads() {
            return this.nthreads;
        }

        public float getQuadDecimate() {
            return this.quadDecimate;
        }

        public float getQuadSigma() {
            return this.quadSigma;
        }

        public boolean getRefineEdges() {
            return this.refineEdges;
        }

        public float getDecodeSharpening() {
            return this.decodeSharpening;
        }

        public Path getDebug() {
            return this.debug;
        }
    }

    public static final class QuadThresholdParameters {
        /** Reject quads containing too few pixels */
        public int minClusterPixels = 5;

        /// How many corner candidates to consider when segmenting a group
        /// of pixels into a quad.
        public int maxNumMaxima = 10;

        /// Reject quads where pairs of edges have angles that are close to
        /// straight or close to 180 degrees. Zero means that no quads are
        /// rejected. (In radians).
        public double criticalAngle = 10 * Math.PI / 180.0;

        /// When fitting lines to the contours, what is the maximum mean
        /// squared error allowed?
        /// This is useful in rejecting contours that are far from being
        /// quad shaped; rejecting these quads "early" saves expensive
        /// decoding processing.
        public float maxLineFitMSE = 10.0f;

        /// When we build our model of black & white pixels, we add an
        /// extra check that the white model must be (overall) brighter
        /// than the black model. 
        /// How much brighter? (in pixel values, [0,255]).
        public int minWhiteBlackDiff = 5;

        /// Should the thresholded image be deglitched?
        /// Only useful for very noisy images
        public boolean deglitch = false;
    }

    private Config params;
    private List<AprilTagFamily> families;

    public AprilTagDetector() {
        this(Collections.emptyList(), new Config());
    }

    public AprilTagDetector(List<AprilTagFamily> families, Config config) {
        super(AprilTagDetector.create());
        this.params = new Config(config);
        this.families = new ArrayList<>(families);
        this.families.sort(Comparator.comparingLong(family -> family.ptr));
    }

    private AprilTagDetector(long ptr) {
        super(ptr);
    }

    // private void updateParams() {
    //     AprilTagDetector.setParams(
    //         this.ptr,
    //         this.params.nthreads,
    //         this.params.quadDecimate,
    //         this.params.quadSigma,
    //         this.params.refineEdges,
    //         this.params.decodeSharpening,
    //         this.params.debug);
    // }

    // public void setThreads(int nthreads) {
    //     this.params.nthreads = nthreads;
    //     this.updateParams();
    // }

    // public int getThreads() {
    //     return this.params.nthreads;
    // }

    // public void setQuadDecimate(float quadDecimate) {
    //     this.params.quadDecimate = quadDecimate;
    //     this.updateParams();
    // }

    // public void addFamily(String familyName) {
        
    // }

    // public void addFamily(String familyName, int maxHamming) {

    // }

    // public void clearFamilies() {
    //     AprilTagDetector.clearFamilies(this.ptr);
    // }

    public AprilTagDetections detect(ByteBuffer buf, int width, int height) throws IllegalArgumentException {
        return this.detect(buf, width, height, width);
    }

    /**
     * Detect AprilTags in image
     * @param buf Data buffer (8-bit grayscale format)
     * @param width Image width (pixels)
     * @param height Image height (pixels)
     * @param stride Image stride (bytes). Must be >= width.
     * @return Detections
     * @throws IllegalArgumentException If the image dimensions are invalid
     * @throws IllegalStateException If this method is called after the detector is closed
     * @throws NullPointerException If the buffer is null
     */
    public AprilTagDetections detect(ByteBuffer buf, int width, int height, int stride) throws IllegalArgumentException, IllegalStateException, NullPointerException {
        Objects.requireNonNull(buf);
        if (width <= 0 || height <= 0)
            throw new IllegalArgumentException("Non-positive dimensions");
        if (stride < width)
            throw new IllegalArgumentException("Stride is smaller than width");

        int expectedCapacity;
        try {
            expectedCapacity = Math.multiplyExact(height, stride);
        } catch (ArithmeticException e) {
            throw new IllegalArgumentException("Capacity too large", e);
        }

        if (buf.remaining() != expectedCapacity)
            throw new IllegalArgumentException("Buffer size is smaller than capacity");
        
        return this.nativeRead(ptr -> AprilTagDetector.detect(ptr, buf, width, height, stride));
    }

    public AprilTegDetections detect(org.opencv.core.Mat img) {
        Objects.requireNonNull(img);

    }

    @Override
    protected native void destroy(long ptr);
}