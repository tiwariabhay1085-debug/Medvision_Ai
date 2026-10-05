import sys
from backend.model.brain_mri.predictor import predict


def main():
    # Check if image path was provided
    if len(sys.argv) < 2:
        print("Please provide an image path.")
        print()
        print("Example:")
        print('python app.py "test_image.jpg"')
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        result = predict(image_path)

        print("\nImage:")
        print(image_path)

        print("\nPrediction:")
        print(result["prediction"])

        print("\nConfidence:")
        print(f'{result["confidence"]}%')

        print("\nAll Probabilities:")
        for cls, prob in result["probabilities"].items():
            print(f"{cls}: {prob}%")

    except FileNotFoundError:
        print(f"\nError: Image not found:")
        print(image_path)

    except Exception as e:
        print(f"\nError while predicting:")
        print(e)


if __name__ == "__main__":
    main()