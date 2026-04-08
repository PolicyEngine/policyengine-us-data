import type { Metadata } from "next";
import "./globals.css";
import Sidebar from "./components/Sidebar";

export const metadata: Metadata = {
  title: "Pipeline explorer — PolicyEngine US Data",
  description: "Interactive pipeline documentation for the US data build process",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full antialiased">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="h-full flex">
        <Sidebar />
        <main className="flex-1 h-full overflow-hidden">{children}</main>
      </body>
    </html>
  );
}
