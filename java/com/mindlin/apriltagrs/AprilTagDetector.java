package com.mindlin.apriltagrs;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.Consumer;
import com.mindlin.apriltagrs.AprilTagLibrary.NativeObject;
import com.mindlin.apriltagrs.AprilTagLibrary.NativeObjectReleasedException;
import com.mindlin.apriltagrs.util.Image2D;

/**
 * A detector for AprilTags
 */
public final class AprilTagDetector extends NativeObject {
    /**
     * Create new detector object
     * 
     * @return Handle to detector
     */
    private static native long nativeCreate(int nthreads, float quadDecimate, float quadSigma, boolean refineEdges, double decodeSharpening, String debugPath, long[] familyPtrs, int[] familyHammings);
    private static native AprilTagDetections nativeDetect(long ptr, ByteBuffer buf, int width, int height, int stride);

    public static class Builder {
        private final Config config;
        private final Map<AprilTagFamily, Integer> families;

        public Builder() {
            this.config = new Config();
            this.families = Collections.emptyMap();
        }

        /**
         * Copy constructor
         */
        public Builder(Builder src) {
            this(Objects.requireNonNull(src).config, src.families);
        }

        private Builder(Config config, Map<AprilTagFamily, Integer> families) {
            Objects.requireNonNull(config, "config");
            Objects.requireNonNull(families, "families");

            for (var entry : families.entrySet()) {
                var family = entry.getKey();
                Objects.requireNonNull(family, "family");
                var bitsCorrected = entry.getValue();
                // Null value == default
                if (bitsCorrected == null)
                    continue;
                
                family.validateBitsCorrected(bitsCorrected.intValue());
            }
            this.config = new Config(config);
            this.families = new LinkedHashMap<>(families);
        }

        // Non-validating constructor
        private Builder(Config config, Map<AprilTagFamily, Integer> families, boolean marker) {
            Objects.requireNonNull(config, "Null config");
            Objects.requireNonNull(families, "Null AprilTag families");
            this.config = config;
            this.families = families;
        }

        public Config getConfig() {
            return this.config;
        }

        private Builder mapConfig(Consumer<Config> update) {
            var config = new Config(this.config);
            update.accept(config);
            return new Builder(config, this.families, true);
        }

        /**
         * Set number of threads to use
         * @param nthreads Number of threads. 0 means 
         * @return modified builder
         */
        public Builder numThreads(int nthreads) {
            return this.mapConfig(config -> config.setNumThreads(nthreads));
        }

        /**
         * Adds an AprilTag family to detect, with a default hamming distance of 2.
         * @param familyName Name of AprilTag family
         * @return Augmented builder
         * @throws NullPointerException If family name is null
         * @throws IllegalArgumentException If family name is invalid
         */
        public Builder addFamily(String familyName) throws NullPointerException, IllegalArgumentException {
            return this.addFamily(familyName, AprilTagFamily.DEFAULT_HAMMING);
        }

        /**
         * Adds an AprilTag family to detect.
         * @param familyName Name of AprilTag family
         * @param bitsCorrected Maxumum number of bits to correct when detecting this family of AprilTags
         * @return
         * @throws NullPointerException If family name is null
         * @throws IllegalArgumentException If family name is invalid, or maxHamming is out of the range [0..3].
         */
        public Builder addFamily(String familyName, int bitsCorrected) throws NullPointerException, IllegalArgumentException {
            Objects.requireNonNull(familyName, "familyName");
            var family = AprilTagFamily.forName(familyName);
            return this.addFamily(family, bitsCorrected);
        }

        public Builder addFamily(AprilTagFamily family) throws IllegalArgumentException {
            return this.addFamily(family, AprilTagFamily.DEFAULT_HAMMING);
        }

        public Builder addFamily(AprilTagFamily family, int bitsCorrected) throws NullPointerException, IllegalArgumentException {
            Objects.requireNonNull(family, "family");
            family.validateBitsCorrected(bitsCorrected);
            var families = new LinkedHashMap<>(this.families);
            families.put(family, bitsCorrected);
            return new Builder(this.config, this.families, true);
        }

        public AprilTagDetector build() {
            // We clone these values here
            var config = this.config.unmodifiable();
            var families = new LinkedHashMap<>(this.families);
            var familyPointerLookup = new HashMap<Long, AprilTagFamily>(this.families.size());

            List<ReentrantReadWriteLock.ReadLock> familyLocks = new ArrayList<>(this.families.size());
            try {
                // Reformat 
                int i = 0;
                long[] familyPtrs = new long[this.families.size()];
                int[] bitsCorrecteds = new int[this.families.size()];
                for (var entry : this.families.entrySet()) {
                    var family = Objects.requireNonNull(entry.getKey());
                    var bitsCorrected = Objects.requireNonNull(entry.getValue()).intValue();
                    family.validateBitsCorrected(bitsCorrected);
                    var lock = family.ptrLock.readLock();
                    lock.lock();
                    try {
                        familyPtrs[i] = family.ptr;
                        bitsCorrecteds[i] = bitsCorrected;
                        familyPointerLookup.put(family.ptr, family);
                        i++;
                        familyLocks.add(lock);
                    } catch (Throwable t) {
                        // Unlock pointer if we won't do this in the finally clause later
                        if (familyLocks.size() == 0 || familyLocks.get(familyLocks.size() - 1) != lock) {
                            lock.unlock();
                        }
                        throw t;
                    }
                }
                assert i == familyPtrs.length;

                long ptr = AprilTagDetector.nativeCreate(
                    config.nthreads,
                    config.quadDecimate,
                    config.quadSigma,
                    config.refineEdges,
                    config.decodeSharpening,
                    config.debug.toAbsolutePath().toString(),
                    familyPtrs,
                    bitsCorrecteds
                );

                return new AprilTagDetector(ptr, config, families, familyPointerLookup);
            } finally {
                // Unlock the pointers for all the AprilTag families
                for (var lock : familyLocks) {
                    lock.unlock();
                }
            }
        }
    }

    /**
     * Configuration for AprilTagDetector
     */
    public static final class Config {
        private boolean mutable = true;
        private int nthreads = 1;
        private float quadDecimate = 2.0f;
        private float quadSigma = 0.0f;
        private boolean refineEdges = true;
        private float decodeSharpening = 0.25f;
        private Path debug = null;

        public Config() {
        }

        /**
         * Copy constructor
         */
        public Config(Config config) {
            Objects.requireNonNull(config, "config");
            this.mutable = true;
            this.nthreads = config.nthreads;
            this.quadDecimate = config.quadDecimate;
            this.quadSigma = config.quadSigma;
            this.refineEdges = config.refineEdges;
            this.decodeSharpening = config.decodeSharpening;
            this.debug = config.debug;
        }

        /**
         * @return An umodifiable copy
         */
        public Config unmodifiable() {
            if (!this.isMutable())
                return this;
            var result = new Config(this);
            result.mutable = false;
            return result;
        }

        public boolean isMutable() {
            return mutable;
        }

        private void assertMutable() throws UnsupportedOperationException {
            if (!this.mutable)
                throw new UnsupportedOperationException("Config is not mutable");
        }

        public int getNumThreads() {
            return this.nthreads;
        }

        public void setNumThreads(int nthreads) throws UnsupportedOperationException {
            this.assertMutable();
            if (nthreads < 0)
                throw new IllegalArgumentException("nthreads must be non-negative");
            this.nthreads = nthreads;
        }

        public float getQuadDecimate() {
            return this.quadDecimate;
        }

        public void setQuadDecimate(float quadDecimate) throws UnsupportedOperationException {
            this.assertMutable();
            this.quadDecimate = quadDecimate;
        }

        public float getQuadSigma() {
            return this.quadSigma;
        }

        public void setQuadSigma(float quadSigma) throws UnsupportedOperationException {
            this.assertMutable();
            this.quadSigma = quadSigma;
        }

        public boolean getRefineEdges() {
            return this.refineEdges;
        }

        public void setRefineEdges(boolean refineEdges) throws UnsupportedOperationException {
            this.assertMutable();
            this.refineEdges = refineEdges;
        }

        public float getDecodeSharpening() {
            return this.decodeSharpening;
        }

        public void setDecodeSharpening(float decodeSharpening) throws UnsupportedOperationException {
            this.assertMutable();
            this.decodeSharpening = decodeSharpening;
        }

        public Path getDebug() {
            return this.debug;
        }

        public void setDebug(Path debug) throws UnsupportedOperationException {
            this.assertMutable();
            this.debug = debug;
        }

        @Override
        public String toString() {
            return "Config [nthreads=" + nthreads + ", quadDecimate=" + quadDecimate + ", quadSigma=" + quadSigma
                    + ", refineEdges=" + refineEdges + ", decodeSharpening=" + decodeSharpening + ", debug=" + debug
                    + "]";
        }

        @Override
        public int hashCode() {
            final int prime = 31;
            int result = 1;
            //TODO: ignore mutable?
            result = prime * result + (mutable ? 1231 : 1237);
            result = prime * result + nthreads;
            result = prime * result + Float.floatToIntBits(quadDecimate);
            result = prime * result + Float.floatToIntBits(quadSigma);
            result = prime * result + (refineEdges ? 1231 : 1237);
            result = prime * result + Float.floatToIntBits(decodeSharpening);
            result = prime * result + Objects.hashCode(debug);
            return result;
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj)
                return true;
            if (obj == null || !(obj instanceof Config))
                return false;
            Config other = (Config) obj;
            return (
                (this.mutable == other.mutable)
                && (this.nthreads == other.nthreads)
                && (Float.floatToIntBits(quadDecimate) == Float.floatToIntBits(other.quadDecimate))
                && (Float.floatToIntBits(quadSigma) == Float.floatToIntBits(other.quadSigma))
                && (refineEdges == other.refineEdges)
                && (Float.floatToIntBits(decodeSharpening) == Float.floatToIntBits(other.decodeSharpening))
                && Objects.equals(this.debug, other.debug)
            );
        }
    }

    public static final class QuadThresholdParameters {
        /** Reject quads containing too few pixels */
        public int minClusterPixels = 5;

        /**
         * How many corner candidates to consider when segmenting a group
         * of pixels into a quad.
         */
        public int maxNumMaxima = 10;

        /**
         * Reject quads where pairs of edges have angles that are close to
         * straight or close to 180 degrees. Zero means that no quads are
         * rejected. (In radians).
         */
        public double criticalAngle = 10 * Math.PI / 180.0;
        /**
         * When fitting lines to the contours, what is the maximum mean
         * squared error allowed?
         * This is useful in rejecting contours that are far from being
         * quad shaped; rejecting these quads "early" saves expensive
         * decoding processing.
         */
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

    /** Configuration */
    private final Config config;
    /** Map of family => hamming distance */
    private final Map<AprilTagFamily, Integer> families;
    /** Map of pointer => family */
    private final Map<Long, AprilTagFamily> ptrLookup;

    private AprilTagDetector(long ptr, Config config, Map<AprilTagFamily, Integer> families, Map<Long, AprilTagFamily> ptrLookup) {
        super(ptr);
        this.config = config;
        this.families = families;
        this.ptrLookup = ptrLookup;
    }

    public AprilTagDetections detect(byte[][] image) throws IllegalArgumentException, NullPointerException, NativeObjectReleasedException {
        var img = Image2D.from(image);
        return this.detect(img);
    }

    public AprilTagDetections detect(ByteBuffer buf, int width, int height) throws IllegalArgumentException, NullPointerException, NativeObjectReleasedException {
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
     * @throws NativeObjectReleasedException If this method is called after the detector is closed
     * @throws NullPointerException If the buffer is null
     */
    public AprilTagDetections detect(ByteBuffer buf, int width, int height, int stride) throws IllegalArgumentException, NativeObjectReleasedException, NullPointerException {
        var image = Image2D.from(buf, width, height, stride);
        
        return this.detect(image);
    }

    public AprilTagDetections detect(Image2D image) {
        Image2D.validate(image);
        var width = image.getWidth();
        var height = image.getHeight();
        var stride = image.getStride();
        var buffer = image.buffer();

        return this.nativeRead(ptr -> AprilTagDetector.nativeDetect(ptr, buffer, width, height, stride));
    }

    @Override
    protected native void destroy(long ptr);
}