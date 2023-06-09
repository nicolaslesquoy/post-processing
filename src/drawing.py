import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from custom_types import Path
import numpy as np
import cv2 


class Drawing:

    def draw_graph_X(title: str, name: str, path_to_folder: Path, result_dict: dict, calibration_positions: dict[str: float], colors: dict, dx: float, yerr: float) -> bool:
        assert list(sorted(result_dict.keys())) == list(sorted(colors.keys()))
        fig, ax = plt.subplots()
        print("Drawing projection along X vs distance")
        for key in result_dict.keys():
            for point in result_dict[key]:
                ax.errorbar(
                    point["distance"],
                    point["coordinates"][0] * dx,
                    yerr=yerr,
                    fmt="--.",
                    color=colors[key],
                    capsize=2
                )
        for key in calibration_positions.keys():
            ax.axvline(x=calibration_positions[key], color="grey", linestyle="--", alpha=0.2)
        # Representation of the plane
        t = patches.Polygon([[9.5, 1.5], [25, 11.8], [25, 1.5]], linewidth=0, hatch="///", fill=None, alpha=0.2)
        p = patches.Rectangle((0,0),width=26.5, height=1.5,linewidth=0, hatch="///", fill=None, alpha=0.2)
        ax.add_patch(p)
        ax.add_patch(t)
        list_patches = []
        if "20" in list(colors.keys()):
            for key in colors.keys():
                list_patches.append(patches.Patch(color=colors[key], label=f"$\\alpha$ = {key}°"))
        elif "5" in list(colors.keys()):
            for key in colors.keys():
                list_patches.append(patches.Patch(color=colors[key], label=f"$\\beta$ = {key}°"))
        else:
            for key in colors.keys():
                list_patches.append(patches.Patch(color=colors[key], label=f"{key}"))
        plt.title(title)
        plt.legend(handles=list_patches)
        plt.xlim(0, 26)
        plt.ylim(0, 9)
        # ax.set_aspect('equal', adjustable='box')
        plt.xlabel("Distance (cm)")
        plt.ylabel("X (cm)")
        plt.savefig(f"{path_to_folder}/{name}_X.png")
        plt.clf()
        return True
    
    def draw_graph_Y(title: str, name: str, path_to_folder: Path, result_dict: dict, calibration_positions: dict[str: float], colors: dict, dy: float, yerr: float) -> bool:
        assert list(sorted(result_dict.keys())) == list(sorted(colors.keys()))
        fig, ax = plt.subplots()
        print("Drawing projection along X vs distance")
        for key in result_dict.keys():
            for point in result_dict[key]:
                ax.errorbar(
                    point["distance"],
                    point["coordinates"][1] * dy,
                    yerr=yerr,
                    fmt="--.",
                    color=colors[key],
                    capsize=2
                )
        for key in calibration_positions.keys():
            ax.axvline(x=calibration_positions[key], color="grey", linestyle="--", alpha=0.2)
        # Representation of the plane
        t = patches.Polygon([[9.5, 1.5], [25, 11.8], [25, 1.5]], linewidth=0, hatch="///", fill=None, alpha=0.2)
        p = patches.Rectangle((0,0),width=26.5, height=1.5,linewidth=0, hatch="///", fill=None, alpha=0.2)
        ax.add_patch(p)
        ax.add_patch(t)
        list_patches = []
        if "20" in list(colors.keys()):
            for key in colors.keys():
                list_patches.append(patches.Patch(color=colors[key], label=f"$\\alpha$ = {key}°"))
        elif "5" in list(colors.keys()):
            for key in colors.keys():
                list_patches.append(patches.Patch(color=colors[key], label=f"$\\beta$ = {key}°"))
        else:
            for key in colors.keys():
                list_patches.append(patches.Patch(color=colors[key], label=f"{key}"))
        plt.title(title)
        plt.legend(handles=list_patches)
        plt.xlim(0, 26)
        plt.ylim(0, 9)
        # ax.set_aspect('equal', adjustable='box')
        plt.xlabel("Distance (cm)")
        plt.ylabel("Y (cm)")
        plt.savefig(f"{path_to_folder}/{name}_Y.png")
        plt.clf()
        return True

    def draw_graph_XY(title: str, name: str, path_to_folder: Path, result_dict: dict, calibration_positions: dict[str: float], colors: dict, dx: float, dy: float, xerr: float, yerr: float) -> bool:
        assert list(sorted(result_dict.keys())) == list(sorted(colors.keys()))
        fig, ax = plt.subplots()
        print("Drawing projection along XY")
        for key in result_dict.keys():
            for point in result_dict[key]:
                ax.errorbar(
                    point["coordinates"][0] * dx,
                    point["coordinates"][1] * dy,
                    xerr=xerr,
                    yerr=yerr,
                    fmt="--.",
                    color=colors[key],
                    capsize=2
                )
        c = patches.Circle((0, 0.8), 1.5, fill=None, alpha=0.2, hatch="///")
        ax.add_patch(c)
        list_patches = []
        if "20" in list(colors.keys()):
            for key in colors.keys():
                list_patches.append(patches.Patch(color=colors[key], label=f"$\\alpha$ = {key}°"))
        elif "5" in list(colors.keys()):
            for key in colors.keys():
                list_patches.append(patches.Patch(color=colors[key], label=f"$\\beta$ = {key}°"))
        else:
            for key in colors.keys():
                list_patches.append(patches.Patch(color=colors[key], label=f"{key}"))
        plt.title(title)
        plt.legend(handles=list_patches)
        plt.xlim(0, 8)
        plt.ylim(0, 4)
        ax.set_aspect('equal', adjustable='box')
        plt.xlabel("X (cm)")
        plt.ylabel("Y (cm)")
        plt.savefig(f"{path_to_folder}/{name}_XY.png")
        plt.clf()
        return True
    
    def draw_stack(path_to_target: Path, name: str, path_to_folder: Path, target_incidence: str, target_derapage: str, incidence: bool):
        number_of_files = len(list(path_to_folder.glob("*.jpg")))
        fraction = 1/number_of_files
        test = cv2.imread(str(list(path_to_folder.glob("*.jpg"))[0]))
        output = np.zeros((test.shape[0], test.shape[1],3))
        for path in path_to_folder.glob("*.jpg"):
            if incidence:
                if target_incidence == path.stem.split("-")[0]:
                    output = np.add(output,fraction*cv2.imread(str(path)))
            else:
                if target_derapage == path.stem.split("-")[1]:
                    output = np.add(output,fraction*cv2.imread(str(path)))

        cv2.imwrite("output.jpg", output)
        img = cv2.imread("output.jpg")
        #  cv2.IMREAD_GRAYSCALE
        img = cv2.rotate(img, cv2.ROTATE_180)
        # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        img = cv2.addWeighted(img, 7, np.zeros(img.shape, img.dtype), 0, 0)
        cv2.imwrite(f"{path_to_target}/{name}.jpg", img)
    
    def draw_comparaison_CFD():
        pass