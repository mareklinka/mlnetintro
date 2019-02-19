using System.IO;
using SkiaSharp;

namespace BA.NET.MLNET.Demo2
{
    public static class ImageUtilities
    {
        public static string ResizeImage(string filename, int targetWidth, int targetHeight, string targetFolder)
        {
            var withoutExt = Path.GetFileNameWithoutExtension(filename);
            var newName = $"{withoutExt}_resized.png";

            using (var original = SKBitmap.Decode(filename))
            using (var resized = original.Resize(new SKImageInfo(targetWidth, targetHeight), SKFilterQuality.None))
            using (var image = SKImage.FromBitmap(resized))
            using (var data = image.Encode(SKEncodedImageFormat.Png, 100))
            using (var stream = new FileStream(Path.Combine(targetFolder, newName), FileMode.Create, FileAccess.Write))
            {
                data.SaveTo(stream);
            }

            return newName;
        }
    }
}