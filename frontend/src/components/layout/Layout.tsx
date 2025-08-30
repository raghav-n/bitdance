import { ReactNode } from "react";
import Navbar, { useNavbar } from "./Navbar";
import Header from "./Header";
import { cn } from "@/lib/utils";

interface LayoutProps {
  children: ReactNode;
  title: string;
}

const Layout = ({ children, title }: LayoutProps) => {
  const { collapsed } = useNavbar();
  
  return (
    <div className="min-h-screen bg-slate-50" style={{ height: '100vh' }}>
      <Navbar />
      <div className={cn(
        "flex flex-col transition-all duration-300",
        collapsed ? "ml-16" : "ml-64"
      )}>
        <Header title={title} />
        <main className="flex-1 p-6 bg-white">
          <div className="max-w-7xl mx-auto">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
};

export default Layout;
