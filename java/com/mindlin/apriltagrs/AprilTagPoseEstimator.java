package com.mindlin.apriltagrs;

import com.mindlin.apriltagrs.AprilTagDetections.Detection;
import com.mindlin.apriltagrs.util.IntArrayList;

import java.util.*;

public final class AprilTagPoseEstimator {
    private static Detection fixref(AprilTagDetection detection) {
        //TODO fixme
        try {
            return (Detection) detection;
        } catch (ClassCastException e) {
            throw new UnsupportedOperationException("Custom AprilTagDetection", e);
        }
    }

    private static native AprilTagPose[] nativeEstimatePoses(
            double tagSize,
            double fx,
            double fy,
            double cx,
            double cy,
            long detectionsPtr,
            int[] idxs
    );

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

    public AprilTagPoseEstimator(Config config) {
        Objects.requireNonNull(config, "config");
        this.tagSize = config.tagsize;
        this.fx = config.fx;
        this.fy = config.fy;
        this.cx = config.cx;
        this.cy = config.cy;
    }

    private AprilTagPose[] estimatePoses(AprilTagDetections detections, int[] indices) {
        int len = (indices == null) ? detections.size() : indices.length;
        if (len == 0)
            return new AprilTagPose[0];
        var result = detections.nativeRead(ptr ->
                AprilTagPoseEstimator.nativeEstimatePoses(
                        this.tagSize,
                        this.fx,
                        this.fy,
                        this.cx,
                        this.cy,
                        ptr,
                        indices
                )
        );
        if (result.length != len)
            throw new IllegalStateException("Result list wrong length");
        return result;
    }

    public AprilTagPose estimatePose(AprilTagDetection detection) {
        var detection1 = fixref(detection);
        AprilTagDetections detections = detection1.getDetections();
        int index = detection1.index;
        AprilTagPose[] result = this.estimatePoses(detections, new int[] { index });
        return result[0];
    }

    private AprilTagPose[] upgradeBuckets(AprilTagDetections dets, IntArrayList idxs0, Detection first, Iterator<? extends AprilTagDetection> iter) {
        class Bucket {
            final IntArrayList detsIdxs;
            final IntArrayList resultIdxs;
            Bucket(IntArrayList detsIdxs) {
                this.detsIdxs = detsIdxs;
                int size = detsIdxs.size();

                int[] resultIdxs = new int[size];
                for (int i = 0; i < size; i++)
                    resultIdxs[i] = i;
                this.resultIdxs = new IntArrayList(resultIdxs);
            }
            Bucket() {
                int capacity = 16;
                this.detsIdxs = new IntArrayList(capacity);
                this.resultIdxs = new IntArrayList(capacity);
            }
            void push(int detsIdx, int resultIdx) {
                this.detsIdxs.push(detsIdx);
                this.resultIdxs.push(resultIdx);
            }
        }

        Map<AprilTagDetections, Bucket> buckets = new HashMap<>();
        int i = idxs0.size();
        {
            // Push first bucket
            buckets.put(dets, new Bucket(idxs0));

            dets = first.getDetections();
            var bucket = new Bucket();
            bucket.push(first.index, i++);
            buckets.put(dets, bucket);

            while (iter.hasNext()) {
                var current = fixref(iter.next());
                if (current.getDetections() != dets)
                    bucket = buckets.computeIfAbsent(dets = current.getDetections(), key -> new Bucket());

                bucket.push(current.index, i++);
            }
        }

        // Now detect stuff
        AprilTagPose[] result = new AprilTagPose[i];
        for (var entry : buckets.entrySet()) {
            var bucket = entry.getValue();
            var chunk = this.estimatePoses(entry.getKey(), bucket.detsIdxs.toIntArray());
            int[] resultIdxs = bucket.resultIdxs.getData();
            for (int j = 0; j < chunk.length; j++) {
                AprilTagPose pose = chunk[j];
                int idx = resultIdxs[j];
                result[idx] = pose;
            }
        }
        return result;
    }

    /**
     * Estimate poses
     * @param detections Apriltag detections
     * @return pose estimations
     */
    public AprilTagPose[] estimatePoses(Collection<? extends AprilTagDetection> detections) {
        int size = detections.size();
        if (size == 0)
            return new AprilTagPose[0];
        
        if (size == 1) {
            var first = fixref(detections.iterator().next());
            return this.estimatePoses(first.getDetections(), new int[] { first.index });
        }
        if (detections instanceof AprilTagDetections) {
            return this.estimatePoses((AprilTagDetections) detections, null);
        }

        // There's a good chance that they're all in the same Detections object
        var iter = detections.iterator();

        AprilTagDetections dets;
        IntArrayList detsIdxs = new IntArrayList(size);
        {
            var first = fixref(iter.next());
            dets = first.getDetections();
            detsIdxs.push(first.index);
        }

        while (iter.hasNext()) {
            var current = fixref(iter.next());
            if (current.getDetections() != dets) {
                return this.upgradeBuckets(dets, detsIdxs, current, iter);
            }
            detsIdxs.push(current.index);
        }
        
        assert detsIdxs.size() == size;

        return this.estimatePoses(dets, detsIdxs.toIntArray());
    }

    public static final class Config {
        public final double tagsize;
        public final double fx;
        public final double fy;
        public final double cx;
        public final double cy;
        public Config(double tagsize, double fx, double fy, double cx, double cy) {
            this.tagsize = tagsize;
            this.fx = fx;
            this.fy = fy;
            this.cx = cx;
            this.cy = cy;
        }
    }
}
