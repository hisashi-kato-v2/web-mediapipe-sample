'use client';

import React, { useEffect, useRef, useState } from 'react';
import { FilesetResolver, ImageSegmenter } from '@mediapipe/tasks-vision';

const legendColors = [
  [255, 197, 0, 255],
  [128, 62, 117, 255],
  [255, 104, 0, 255],
  [166, 189, 215, 255],
  [193, 0, 32, 255],
  [206, 162, 98, 255],
  [129, 112, 102, 255],
  [0, 125, 52, 255],
  [246, 118, 142, 255],
  [0, 83, 138, 255],
  [255, 112, 92, 255],
  [83, 55, 112, 255],
  [255, 142, 0, 255],
  [179, 40, 81, 255],
  [244, 200, 0, 255],
  [127, 24, 13, 255],
  [147, 170, 0, 255],
  [89, 51, 21, 255],
  [241, 58, 19, 255],
  [35, 44, 22, 255],
  [0, 161, 194, 255],
];

const MediaPipeSegmenter: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null); // ウェブカメラの映像を表示するためのビデオ要素
  const canvasRef = useRef<HTMLCanvasElement>(null); // セグメンテーション後の画像を表示するためのキャンバス
  const [imageSegmenter, setImageSegmenter] = useState<ImageSegmenter | null>(
    null
  );
  const webcamRunningRef = useRef(false); // ウェブカメラが動いているかどうか
  const [runningMode, setRunningMode] = useState<'IMAGE' | 'VIDEO'>('IMAGE'); // セグメンテーションの実行モード
  useEffect(() => {
    const loadModel = async () => {
      const fileset = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2/wasm'
      );
      const segmenter = await ImageSegmenter.createFromOptions(fileset, {
        baseOptions: {
          modelAssetPath:
            'https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite',
          delegate: 'GPU',
        },
        runningMode: runningMode,
        outputCategoryMask: true,
        outputConfidenceMasks: false,
      });
      setImageSegmenter(segmenter);
    };
    loadModel();
  }, []);

  const toggleWebcam = async () => {
    if (!imageSegmenter) return;
    if (webcamRunningRef.current) {
      webcamRunningRef.current = false;
      return;
    }

    const constraints = { video: true };
    const stream = await navigator.mediaDevices.getUserMedia(constraints);

    // ビデオを再生
    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      videoRef.current.play();
    }

    webcamRunningRef.current = true;

    // ビデオからフレームを取得してセグメンテーションを実行
    const processWebcam = async () => {
      if (!canvasRef.current || !videoRef.current || !webcamRunningRef.current)
        return;
      const nowVideo = videoRef.current; // 現在写しているビデオ要素
      const refContext = canvasRef.current.getContext('2d');
      if (!refContext) return;

      // 動画のメタデータが読み込まれるまで待つ
      if (nowVideo.videoWidth === 0 || nowVideo.videoHeight === 0) {
        console.warn('Video dimensions are not ready yet. Waiting...');
        await new Promise((resolve) => {
          nowVideo.addEventListener('loadedmetadata', resolve, { once: true });
        });
      }

      // 再度チェック
      if (nowVideo.videoWidth === 0 || nowVideo.videoHeight === 0) {
        console.error('Video dimensions are still 0. Skipping frame.');
        return;
      }

      // キャンバスにビデオのフレームを描画(セグメンテーションのため)
      refContext.drawImage(
        videoRef.current,
        0,
        0,
        videoRef.current.videoWidth,
        videoRef.current.videoHeight
      );
      if (runningMode === 'IMAGE') {
        setRunningMode('VIDEO');
        await imageSegmenter.setOptions({ runningMode: 'VIDEO' });
      }

      const startTimeMs = performance.now();
      imageSegmenter.segmentForVideo(
        videoRef.current,
        startTimeMs,
        (result) => {
          if (!refContext) return;
          const imageData = refContext.getImageData(
            0,
            0,
            videoRef.current?.videoWidth || 0,
            videoRef.current?.videoHeight || 0
          );
          const mask = result.categoryMask?.getAsFloat32Array();

          if (!mask) return;
          let j = 0;
          for (let i = 0; i < mask.length; i++) {
            // 赤いところが人体。それ以外は黄色
            const color =
              legendColors[Math.round(mask[i] * 255) % legendColors.length];
            imageData.data[j] = (color[0] + imageData.data[j]) / 2;
            imageData.data[j + 1] = (color[1] + imageData.data[j + 1]) / 2;
            imageData.data[j + 2] = (color[2] + imageData.data[j + 2]) / 2;
            imageData.data[j + 3] = (color[3] + imageData.data[j + 3]) / 2;
            j += 4;
          }
          // キャンバスにセグメンテーション結果を描画
          refContext.putImageData(imageData, 0, 0);
          if (webcamRunningRef.current) requestAnimationFrame(processWebcam);
        }
      );
    };
    requestAnimationFrame(processWebcam);
  };

  return (
    <>
      <button onClick={toggleWebcam}>
        {webcamRunningRef.current ? 'DISABLE WEBCAM' : 'ENABLE WEBCAM'}
      </button>
      <video ref={videoRef} width='1280px' height='720px' />
      <canvas ref={canvasRef} width='1280px' height='720px'></canvas>
    </>
  );
};

export default MediaPipeSegmenter;
