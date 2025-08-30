import { createContext, useContext } from "react";
import { Link, useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  Home,
  ChevronLeft,
  ChevronRight,
  BarChart3,
  Brain,
  Zap,
  Shield,
} from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

// Create a context for the navbar state
export const NavbarContext = createContext({
  collapsed: false,
  toggleCollapsed: () => {},
});

// Custom hook to use the navbar context
export const useNavbar = () => useContext(NavbarContext);

const Navbar = () => {
  const { collapsed, toggleCollapsed } = useNavbar();
  const location = useLocation();

  // Only the 5 dashboard navigation items
  const navItems = [
    { icon: Home, label: "Overview Dashboard", path: "/" },
    { icon: BarChart3, label: "Analytics", path: "/analytics" },
    { icon: Brain, label: "Classification", path: "/classification" },
    { icon: Zap, label: "Prediction", path: "/prediction" },
    { icon: Shield, label: "Violations", path: "/violations" },
  ];

  return (
    <TooltipProvider>
      <nav
        className={cn(
          "bg-[#040F3D] text-white flex flex-col transition-all duration-300 fixed top-0 left-0 bottom-0",
          collapsed ? "w-16" : "w-64"
        )}
      >
        {/* Header */}
        <div
          className={cn(
            "flex items-center p-4 flex-shrink-0",
            collapsed ? "justify-center" : "justify-between",
          )}
        >
          {!collapsed && <span className="font-bold text-xl">🕺BitDance</span>}
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleCollapsed}
            className="text-white hover:bg-[#0A1956]"
          >
            {collapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
          </Button>
        </div>

        {/* Navigation Items */}
        <div className="flex-1 px-3 py-4 overflow-y-auto">
          <ul className="space-y-2">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path;

              const navItem = (
                <Link
                  to={item.path}
                  className={cn(
                    "flex items-center px-3 py-2 rounded-lg transition-colors",
                    isActive
                      ? "bg-[#0A1956] text-white"
                      : "text-gray-300 hover:bg-[#0A1956] hover:text-white",
                    collapsed ? "justify-center" : "justify-start"
                  )}
                >
                  <Icon className="h-5 w-5" />
                  {!collapsed && (
                    <span className="ml-3 text-sm font-medium">{item.label}</span>
                  )}
                </Link>
              );

              if (collapsed) {
                return (
                  <Tooltip key={item.path}>
                    <TooltipTrigger asChild>
                      <li>{navItem}</li>
                    </TooltipTrigger>
                    <TooltipContent side="right">
                      <p>{item.label}</p>
                    </TooltipContent>
                  </Tooltip>
                );
              }

              return <li key={item.path}>{navItem}</li>;
            })}
          </ul>
        </div>
      </nav>
    </TooltipProvider>
  );
};

export default Navbar;
