import { ReactNode, useState } from "react";
import { NavbarContext } from "./Navbar";

interface NavbarProviderProps {
  children: ReactNode;
}

export const NavbarProvider = ({ children }: NavbarProviderProps) => {
  const [collapsed, setCollapsed] = useState(false);

  const toggleCollapsed = () => {
    setCollapsed(!collapsed);
  };

  return (
    <NavbarContext.Provider value={{ collapsed, toggleCollapsed }}>
      {children}
    </NavbarContext.Provider>
  );
};

export default NavbarProvider; 