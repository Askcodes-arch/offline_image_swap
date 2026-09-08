using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

namespace OfflineImageSwapLauncher
{
    internal static class Program
    {
        [STAThread]
        private static void Main()
        {
            string baseDir =
                AppDomain.CurrentDomain.BaseDirectory;

            string python =
                Path.Combine(
                    baseDir,
                    "runtime",
                    "python.exe"
                );

            string script =
                Path.Combine(
                    baseDir,
                    "main.py"
                );

            if (!File.Exists(python))
            {
                MessageBox.Show(
                    "OfflineImageSwap runtime was not found.\n\n" +
                    "Please install the runtime package first.",
                    "Offline Image Swap",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );

                return;
            }

            if (!File.Exists(script))
            {
                MessageBox.Show(
                    "main.py was not found.",
                    "Offline Image Swap",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );

                return;
            }

            try
            {
                ProcessStartInfo startInfo =
                    new ProcessStartInfo();

                startInfo.FileName = python;
                startInfo.Arguments =
                    "\"" + script + "\"";

                startInfo.WorkingDirectory =
                    baseDir;

                startInfo.UseShellExecute =
                    false;

                startInfo.CreateNoWindow =
                    true;

                Process.Start(
                    startInfo
                );
            }
            catch (Exception error)
            {
                MessageBox.Show(
                    error.Message,
                    "Offline Image Swap",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
            }
        }
    }
}
