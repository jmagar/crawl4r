#!/bin/bash
# Test different parallel batch configurations

echo "=========================================================================="
echo "Testing Parallel Batch Configurations"
echo "=========================================================================="
echo ""

# Baseline: Current configuration (96 batch, 1 parallel)
echo "TEST 1: Baseline - 96 batch size, 1 parallel (sequential)"
echo "=========================================================================="
python examples/stress_test_pipeline.py \
    --batch-size 96 \
    --parallel-batches 1 \
    --concurrency 30 \
    | tee /tmp/test1_baseline.log
echo ""
echo ""

# Test 2: 48 batch size, 2 parallel (safer)
echo "TEST 2: Parallel Small - 48 batch size, 2 parallel (total 96)"
echo "=========================================================================="
python examples/stress_test_pipeline.py \
    --batch-size 48 \
    --parallel-batches 2 \
    --concurrency 30 \
    | tee /tmp/test2_48x2.log
echo ""
echo ""

# Test 3: 32 batch size, 3 parallel (more aggressive)
echo "TEST 3: Parallel Medium - 32 batch size, 3 parallel (total 96)"
echo "=========================================================================="
python examples/stress_test_pipeline.py \
    --batch-size 32 \
    --parallel-batches 3 \
    --concurrency 30 \
    | tee /tmp/test3_32x3.log
echo ""
echo ""

# Test 4: 24 batch size, 4 parallel (aggressive)
echo "TEST 4: Parallel Large - 24 batch size, 4 parallel (total 96)"
echo "=========================================================================="
python examples/stress_test_pipeline.py \
    --batch-size 24 \
    --parallel-batches 4 \
    --concurrency 30 \
    | tee /tmp/test4_24x4.log
echo ""
echo ""

# Summary
echo "=========================================================================="
echo "SUMMARY"
echo "=========================================================================="
echo ""
echo "Extracting total times..."
echo ""
grep "Total time:" /tmp/test1_baseline.log | tail -1
grep "Total time:" /tmp/test2_48x2.log | tail -1
grep "Total time:" /tmp/test3_32x3.log | tail -1
grep "Total time:" /tmp/test4_24x4.log | tail -1
echo ""
echo "Full logs in /tmp/test*.log"
