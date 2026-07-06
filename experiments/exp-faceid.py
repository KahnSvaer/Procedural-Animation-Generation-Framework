import sys
sys.path.append(".")  # Add animgen to path

from animgen.core.generated_asset_class import GeneratedAssetClass

from PIL import Image

SAMPLE_OBJECT = "./generated_data/models/paint_mesh_Dolphin.glb"

if __name__ == "__main__":
    source2 = GeneratedAssetClass(SAMPLE_OBJECT)

    views_output = source2.views_output
    for k,v in views_output.items():
        print(f"{k}: {len(v)}")


    for k in range(len(views_output['matte'])):
        views_output['matte'][k].save(f"temp/matte_{k}.png")
        