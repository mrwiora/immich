import { AssetTypeEnum, type AssetFaceResponseDto } from '@immich/sdk';
import type { Faces } from '$lib/managers/asset-viewer-manager.svelte';
import { getAssetMediaUrl, getAssetPlaybackUrl } from '$lib/utils';
import { mapNormalizedRectToContent, type Rect, type Size } from '$lib/utils/container-utils';

export type BoundingBox = Rect & { id: string; labelWidth: number };

export const getBoundingBox = (faces: Faces[], imageSize: Size): BoundingBox[] => {
  const boxes: BoundingBox[] = [];

  for (const face of faces) {
    const rect = mapNormalizedRectToContent(
      { x: face.boundingBoxX1 / face.imageWidth, y: face.boundingBoxY1 / face.imageHeight },
      { x: face.boundingBoxX2 / face.imageWidth, y: face.boundingBoxY2 / face.imageHeight },
      imageSize,
    );

    boxes.push({ id: face.id, ...rect, labelWidth: rect.width });
  }

  return boxes;
};

const loadVideoFrame = async (assetId: string, timestamp: number): Promise<HTMLVideoElement | undefined> => {
  const video = document.createElement('video');

  const seeked = await new Promise<boolean>((resolve) => {
    video.addEventListener(
      'loadedmetadata',
      () => {
        video.currentTime = timestamp / 1000;
      },
      { once: true },
    );
    video.addEventListener('seeked', () => resolve(true), { once: true });
    video.addEventListener('error', () => resolve(false), { once: true });
    video.src = getAssetPlaybackUrl({ id: assetId });
  });

  return seeked ? video : undefined;
};

const loadImage = async (src: string) => {
  const image = new Image();
  image.src = src;

  await new Promise<void>((resolve) => {
    image.addEventListener('load', () => resolve());
    image.addEventListener('error', () => resolve());
  });

  return image;
};

export const zoomImageToBase64 = async (
  face: AssetFaceResponseDto,
  assetId: string,
  assetType: AssetTypeEnum,
  photoViewer: HTMLImageElement | undefined,
): Promise<string | null> => {
  let source: { element: CanvasImageSource; width: number; height: number } | undefined;
  if (assetType === AssetTypeEnum.Video && face.frameTimestamp !== null && face.frameTimestamp !== undefined) {
    // the face was detected in a frame of the video instead of its preview
    const video = await loadVideoFrame(assetId, face.frameTimestamp);
    if (video) {
      source = { element: video, width: video.videoWidth, height: video.videoHeight };
    }
  } else {
    let image: HTMLImageElement | undefined;
    if (assetType === AssetTypeEnum.Image) {
      image = photoViewer;
    } else if (assetType === AssetTypeEnum.Video) {
      image = await loadImage(getAssetMediaUrl({ id: assetId }));
    }

    if (image) {
      source = { element: await loadImage(image.src), width: image.naturalWidth, height: image.naturalHeight };
    }
  }

  if (!source) {
    return null;
  }
  const { boundingBoxX1: x1, boundingBoxX2: x2, boundingBoxY1: y1, boundingBoxY2: y2, imageWidth, imageHeight } = face;

  const coordinates = {
    x1: (source.width / imageWidth) * x1,
    x2: (source.width / imageWidth) * x2,
    y1: (source.height / imageHeight) * y1,
    y2: (source.height / imageHeight) * y2,
  };

  const faceWidth = coordinates.x2 - coordinates.x1;
  const faceHeight = coordinates.y2 - coordinates.y1;

  const canvas = document.createElement('canvas');
  canvas.width = faceWidth;
  canvas.height = faceHeight;

  const context = canvas.getContext('2d');
  if (!context) {
    return null;
  }
  context.drawImage(source.element, coordinates.x1, coordinates.y1, faceWidth, faceHeight, 0, 0, faceWidth, faceHeight);
  return canvas.toDataURL();
};
