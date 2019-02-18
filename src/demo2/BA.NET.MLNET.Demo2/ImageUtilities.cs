using System.IO;
using SkiaSharp;

namespace BA.NET.MLNET.Demo2
{
    public static class ImageUtilities
    {
        public static (string NewName, int OriginalWidth, int OriginalHeight) FlipRotateImage(string filename, int targetWidth, int targetHeight, string targetFolder)
        {
            var withoutExt = Path.GetFileNameWithoutExtension(filename);

            using (var original = SKBitmap.Decode(filename))
            using (var image = SKImage.FromBitmap(Resize(original, targetWidth, targetHeight)))
            {
                var newName = $"{withoutExt}_resized.png";
                var data = image.Encode(SKEncodedImageFormat.Png, 100);

                using (var stream = new FileStream(Path.Combine(targetFolder, newName), FileMode.Create,
                    FileAccess.Write))
                {
                    data.SaveTo(stream);
                }

                return (newName, original.Width, original.Height);
            }
        }

        public static SKBitmap Resize(SKBitmap original, int targetWidth, int targetHeight)
        {
            var resized = original.Resize(new SKImageInfo(targetWidth, targetHeight), SKFilterQuality.None);

            return resized;
        }
    }
}