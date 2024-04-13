package ai_ml;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.CvType;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.VideoWriter;
import org.opencv.videoio.Videoio;
import org.opencv.objdetect.CascadeClassifier;

public class AdvertisementInsertion {

	static {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	}

	public static void main(String[] args) {
		// Load input video
		VideoCapture videoCapture = new VideoCapture("input_video.mp4");
		// Load input image
		Mat advertisementImage = Imgcodecs.imread("input_image.jpg", Imgcodecs.IMREAD_UNCHANGED);

		// Create VideoWriter object for output video
		VideoWriter videoWriter = new VideoWriter("output_video.mp4", VideoWriter.fourcc('H', '2', '6', '4'),
				videoCapture.get(Videoio.CAP_PROP_FPS), new Size((int) videoCapture.get(Videoio.CAP_PROP_FRAME_WIDTH),
						(int) videoCapture.get(Videoio.CAP_PROP_FRAME_HEIGHT)),
				true);

		// Loop through video frames
		Mat frame = new Mat();
		while (videoCapture.read(frame)) {
			// Assuming hand detection and insertion point calculation here
			Rect handPosition = detectHand(frame);
			Point insertionPoint = calculateInsertionPoint(frame, handPosition);

			// Insert advertisement into frame
			Point insertionPoint1 = new Point(100, 100); // Placeholder insertion point
			insertAdvertisement(frame, advertisementImage, insertionPoint1);

			// Write modified frame to output video
			videoWriter.write(frame);

			// Display frame with inserted advertisement
			HighGui.imshow("Video with Advertisement", frame);
			HighGui.waitKey(25);

			if (HighGui.waitKey(25) == 'q') {
				break;
			}
		}

		// Release video capture, writer, and close windows
		videoCapture.release();
		videoWriter.release();
		HighGui.destroyAllWindows();
	}

	// Function to detect hand region in a video frame
	private static Rect detectHand(Mat frame) {
		// Convert frame to grayscale
		Mat grayFrame = new Mat();
		Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);

		// Apply Gaussian blur to reduce noise
		Imgproc.GaussianBlur(grayFrame, grayFrame, new Size(5, 5), 0);

		// Perform hand detection using a pre-trained Haar cascade classifier
		CascadeClassifier handCascade = new CascadeClassifier("haarcascade_hand.xml");
		MatOfRect hands = new MatOfRect();
		handCascade.detectMultiScale(grayFrame, hands);

		// If no hands detected, return an empty rectangle
		if (hands.toArray().length == 0) {
			return new Rect();
		}

		// Get the bounding box of the first detected hand
		Rect handBoundingBox = hands.toArray()[0];

		// Return the bounding box of the detected hand region
		return handBoundingBox;
	}

	// Function to calculate insertion point based on hand position
	private static Point calculateInsertionPoint(Mat frame, Rect handPosition) {
		// Calculate the center of the hand bounding box
		int handCenterX = handPosition.x + handPosition.width / 2;
		int handCenterY = handPosition.y + handPosition.height / 2;

		// Calculate insertion point relative to the hand center
		// For example, place the advertisement slightly above and to the right of the
		// hand
		int insertionX = handCenterX + 50;
		int insertionY = handCenterY - 50;

		// Ensure insertion point is within the frame boundaries
		insertionX = Math.max(0, Math.min(frame.cols() - 1, insertionX));
		insertionY = Math.max(0, Math.min(frame.rows() - 1, insertionY));

		// Return the calculated insertion point
		return new Point(insertionX, insertionY);
	}

	// Function to insert advertisement image into video frame
	private static void insertAdvertisement(Mat videoFrame, Mat advertisementImage, Point insertionPoint) {
		// Get dimensions of the advertisement image
		int adHeight = advertisementImage.rows();
		int adWidth = advertisementImage.cols();

		// Define ROI (Region of Interest) for advertisement insertion
		Rect roi = new Rect(insertionPoint, new Size(adWidth, adHeight));

		// Extract ROI from video frame
		Mat roiMat = videoFrame.submat(roi);

		// Blend advertisement with ROI using the alpha channel
		for (int y = 0; y < adHeight; y++) {
			for (int x = 0; x < adWidth; x++) {
				double[] pixel = advertisementImage.get(y, x);
				double alpha = pixel[3] / 255.0; // Normalize alpha value
				for (int c = 0; c < 3; c++) {
					roiMat.put(y, x, c, (1 - alpha) * roiMat.get(y, x)[c] + alpha * pixel[c]);
				}
			}
		}
	}
}
