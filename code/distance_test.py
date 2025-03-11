from picarx import Picarx
import time
import sys


def main():
    try:
        px = Picarx()
        # px = Picarx(ultrasonic_pins=['D2','D3']) # tring, echo

        while True:
            distance = round(px.ultrasonic.read(), 2)
            print("distance: ",distance)
            time.sleep(1)

    finally:
        px.stop()
        sys.exit(0)


if __name__ == "__main__":
    main()